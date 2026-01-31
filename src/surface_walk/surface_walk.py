#!/usr/bin/env python3
import argparse, json, math, os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from Bio.PDB import PDBParser, is_aa

# --- Optional: AlphaFold pLDDT handling ---
def extract_plddt_from_bfactor(residue) -> float:
    # AlphaFold stores pLDDT in B-factor field in the PDB.
    # Use CA atom if present.
    if residue.has_id("CA"):
        return float(residue["CA"].get_bfactor())
    # fall back: mean B-factor over atoms
    bfs = [a.get_bfactor() for a in residue.get_atoms()]
    return float(np.mean(bfs)) if bfs else float("nan")

AA3_TO_1 = {
    "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H","ILE":"I","LYS":"K","LEU":"L",
    "MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R","SER":"S","THR":"T","VAL":"V","TRP":"W","TYR":"Y"
}

# Maximum solvent accessible surface area by residue (Tien et al.-style values, Å^2)
# Good enough for RSA normalization; swap table later if you prefer.
MAX_SASA = {
    "A":129.0,"C":167.0,"D":193.0,"E":223.0,"F":240.0,"G":104.0,"H":224.0,"I":197.0,"K":236.0,"L":201.0,
    "M":224.0,"N":195.0,"P":159.0,"Q":225.0,"R":274.0,"S":155.0,"T":172.0,"V":174.0,"W":285.0,"Y":263.0
}

def sidechain_centroid(residue):
    # centroid of side-chain heavy atoms; for Gly use CA
    atoms = []
    for atom in residue.get_atoms():
        name = atom.get_name()
        if name in ("N","CA","C","O"):  # exclude backbone
            continue
        if atom.element == "H":  # ignore hydrogens if present
            continue
        atoms.append(atom.get_coord())
    if not atoms:
        return residue["CA"].get_coord() if residue.has_id("CA") else None
    return np.mean(np.vstack(atoms), axis=0)

def compute_sasa_with_freesasa(pdb_path: str, chain: str):
    """
    Compute per-residue SASA using freesasa.
    Returns dict keyed by (chain_id, resseq, icode) -> sasa
    """
    import freesasa
    structure = freesasa.Structure(pdb_path)
    result = freesasa.calc(structure)
    # freesasa residue areas are accessible via result.residueAreas()
    ra = result.residueAreas()
    sasa_map = {}
    for ch in ra:
        if chain and ch != chain:
            continue
        for resn in ra[ch]:
            # resn is residue number as string; may include insertion code in some cases
            # freesasa doesn't always expose insertion codes cleanly; we use best-effort parsing
            # If needed, align by Biopython residues later.
            areas = ra[ch][resn]
            # total = areas.total
            try:
                resseq = int(''.join([c for c in resn if c.isdigit() or c == '-']))
            except Exception:
                continue
            sasa_map[(ch, resseq, "")] = float(areas.total)
    return sasa_map

def build_residue_table(pdb_path: str, chain_id: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)
    model = next(structure.get_models())
    if chain_id:
        if chain_id not in model:
            raise KeyError(f"Chain {chain_id!r} not found in PDB")
        chain = model[chain_id]
    else:
        chain = next(model.get_chains())
        chain_id = chain.id

    # SASA
    sasa_map = compute_sasa_with_freesasa(pdb_path, chain_id)

    rows = []
    for res in chain.get_residues():
        if not is_aa(res, standard=True):
            continue
        resname = res.get_resname().upper()
        if resname not in AA3_TO_1:
            continue
        aa = AA3_TO_1[resname]
        hetflag, resseq, icode = res.get_id()
        icode = icode.strip() if isinstance(icode, str) else ""
        if not res.has_id("CA"):
            continue
        ca = res["CA"].get_coord()
        sc = sidechain_centroid(res)
        plddt = extract_plddt_from_bfactor(res)
        sasa = sasa_map.get((chain_id, resseq, icode), float("nan"))
        if math.isnan(sasa):
            # freesasa doesn't expose insertion codes reliably; fall back to blank icode
            sasa = sasa_map.get((chain_id, resseq, ""), float("nan"))
        rsa = float("nan")
        if not math.isnan(sasa) and aa in MAX_SASA:
            rsa = min(1.0, max(0.0, sasa / MAX_SASA[aa]))

        sc_x = float(sc[0]) if sc is not None else float("nan")
        sc_y = float(sc[1]) if sc is not None else float("nan")
        sc_z = float(sc[2]) if sc is not None else float("nan")
        rows.append(dict(
            chain=chain_id,
            resseq=int(resseq),
            icode=icode,
            aa=aa,
            ca_x=float(ca[0]), ca_y=float(ca[1]), ca_z=float(ca[2]),
            sc_x=sc_x, sc_y=sc_y, sc_z=sc_z,
            sasa=sasa, rsa=rsa, pLDDT=plddt
        ))
    df = pd.DataFrame(rows)
    df.insert(0, "res_id", np.arange(len(df), dtype=int))
    return df

def build_surface_graph(df: pd.DataFrame, rsa_thr: float, dcut: float, sigma: float):
    df = df.copy()
    df["is_surface"] = (df["rsa"].fillna(0.0) > rsa_thr)

    surf = df[df["is_surface"]].copy()
    if surf.empty:
        raise RuntimeError("No surface residues found; lower rsa_thr or check SASA computation.")

    coords = surf[["sc_x","sc_y","sc_z"]].to_numpy()
    # Handle any NaNs (should be rare). Fallback to CA coords if needed.
    bad = np.isnan(coords).any(axis=1)
    if bad.any():
        coords[bad] = surf.loc[bad, ["ca_x","ca_y","ca_z"]].to_numpy()

    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=dcut)  # set of (i,j) indices into surf
    edges = []
    for a,b in pairs:
        pa, pb = coords[a], coords[b]
        dist = float(np.linalg.norm(pa - pb))
        w = math.exp(-(dist*dist)/(2*sigma*sigma))
        edges.append((int(surf.iloc[a]["res_id"]), int(surf.iloc[b]["res_id"]), dist, w))

    G = nx.Graph()
    # add nodes with attributes
    for _, row in surf.iterrows():
        G.add_node(int(row["res_id"]))
    # add edges
    for i,j,dist,w in edges:
        G.add_edge(i,j, dist=dist, w=w)

    # connected components
    comp_map = {}
    for cid, comp in enumerate(nx.connected_components(G)):
        for n in comp:
            comp_map[n] = cid
    df["comp_id"] = df["res_id"].map(comp_map).astype("Int64")
    return df, G, pd.DataFrame(edges, columns=["i","j","dist","w"])

def umap_embed_from_graph(G: nx.Graph, n_neighbors=15, min_dist=0.05, random_state=0):
    import umap
    nodes = sorted(G.nodes())
    idx = {n:k for k,n in enumerate(nodes)}
    # Build sparse weighted adjacency
    rows, cols, data = [], [], []
    for u,v,attr in G.edges(data=True):
        w = float(attr.get("w", 1.0))
        iu, iv = idx[u], idx[v]
        rows += [iu, iv]
        cols += [iv, iu]
        data += [w, w]
    A = csr_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes)))
    # Convert to shortest-path distances on the graph using edge length = dist (not weight)
    # We'll build a length matrix from 'dist' attributes
    rows, cols, data = [], [], []
    for u,v,attr in G.edges(data=True):
        d = float(attr.get("dist", 1.0))
        iu, iv = idx[u], idx[v]
        rows += [iu, iv]
        cols += [iv, iu]
        data += [d, d]
    L = csr_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes)))
    D = shortest_path(L, directed=False, unweighted=False)
    # Replace inf with large value (disconnected shouldn't happen if you embed per component;
    # but we keep it safe)
    finite = np.isfinite(D)
    maxd = D[finite].max() if finite.any() else 1.0
    D[~finite] = maxd * 2.0

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="precomputed",
        random_state=random_state
    )
    XY = reducer.fit_transform(D)
    emb = pd.DataFrame({"res_id": nodes, "umap_x": XY[:,0], "umap_y": XY[:,1]})
    return emb

def surface_walk(G: nx.Graph, df: pd.DataFrame, component_id: int, use_2d=None):
    """
    Deterministic traversal within one connected component.
    use_2d: dict res_id -> (x,y) for jump distance tie-breaking (optional)
    """
    nodes = [n for n in G.nodes() if int(df.loc[df.res_id == n, "comp_id"].iloc[0]) == component_id]
    if not nodes:
        return None

    sub = G.subgraph(nodes).copy()
    # start: minimum resseq, tie-break by max rsa
    sub_df = df[df["res_id"].isin(nodes)].copy()
    start = sub_df.sort_values(["resseq", "rsa"], ascending=[True, False]).iloc[0]["res_id"]
    start = int(start)

    visited = set([start])
    path = [start]
    jumps = []  # indices where a jump happened

    def nearest_unvisited_neighbor(cur):
        cand = []
        for nb in sub.neighbors(cur):
            if nb in visited:
                continue
            cand.append((sub[cur][nb].get("dist", 1e9), nb))
        if not cand:
            return None
        cand.sort()
        return cand[0][1]

    def jump_to_next(cur):
        remaining = [n for n in sub.nodes() if n not in visited]
        if not remaining:
            return None
        if use_2d is None:
            # fallback: pick remaining with smallest graph distance (should be inf only if disconnected)
            # Here component is connected, so shortest path exists; we can approximate by euclidean in 3D
            cur_xyz = df.loc[df.res_id == cur, ["sc_x","sc_y","sc_z"]].to_numpy(dtype=float)[0]
            rem_xyz = df[df.res_id.isin(remaining)][["sc_x","sc_y","sc_z"]].to_numpy(dtype=float)
            d = np.linalg.norm(rem_xyz - cur_xyz[None,:], axis=1)
            return int(df[df.res_id.isin(remaining)].iloc[int(np.argmin(d))]["res_id"])
        else:
            cx, cy = use_2d[cur]
            best = None
            bestd = 1e18
            for n in remaining:
                x,y = use_2d[n]
                d = (x-cx)**2 + (y-cy)**2
                if d < bestd:
                    bestd = d
                    best = n
            return int(best)

    cur = start
    while len(visited) < sub.number_of_nodes():
        nxt = nearest_unvisited_neighbor(cur)
        if nxt is None:
            nxt = jump_to_next(cur)
            if nxt is None:
                break
            jumps.append(len(path))  # next index is a jump
        visited.add(nxt)
        path.append(nxt)
        cur = nxt

    aas = df.set_index("res_id").loc[path, "aa"].astype(str).tolist()
    surface_string = "".join(aas)
    return {"component": component_id, "start": start, "path": path, "jumps": jumps, "surface_string": surface_string}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--chain", default="A")
    ap.add_argument("--rsa_thr", type=float, default=0.0)
    ap.add_argument("--dcut", type=float, default=7.5)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--embed", action="store_true", help="Run UMAP embedding")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = build_residue_table(args.pdb, args.chain)
    df2, G, edges_df = build_surface_graph(df, args.rsa_thr, args.dcut, args.sigma)

    # embedding (per component is better, but start with whole graph)
    if args.embed:
        emb = umap_embed_from_graph(G)
        df2 = df2.merge(emb, on="res_id", how="left")
        use_2d = {int(r.res_id):(float(r.umap_x), float(r.umap_y)) for r in emb.itertuples()}
    else:
        use_2d = None

    # write residues and edges
    df2.to_csv(os.path.join(args.outdir, "residues.tsv"), sep="\t", index=False)
    edges_df.to_csv(os.path.join(args.outdir, "edges.tsv"), sep="\t", index=False)

    # traversals per component
    surf_nodes = [n for n in G.nodes()]
    comps = sorted(set(df2[df2.res_id.isin(surf_nodes)]["comp_id"].dropna().astype(int).tolist()))
    walks = []
    for cid in comps:
        w = surface_walk(G, df2, cid, use_2d=use_2d)
        if w:
            walks.append(w)
    with open(os.path.join(args.outdir, "walks.json"), "w") as f:
        json.dump(walks, f, indent=2)

    meta = {
        "pdb": args.pdb,
        "chain": args.chain,
        "rsa_thr": args.rsa_thr,
        "dcut": args.dcut,
        "sigma": args.sigma,
        "embed": bool(args.embed),
        "notes": "Surface residues graph + optional UMAP embedding + greedy surface walk per component."
    }
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote: {args.outdir}/residues.tsv, edges.tsv, walks.json, meta.json")
    if walks:
        print("Example surface string (component 0):")
        print(walks[0]["surface_string"][:120] + ("..." if len(walks[0]["surface_string"])>120 else ""))

if __name__ == "__main__":
    main()
