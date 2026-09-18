import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from nova.equilibrium.flux_surface_connectivity import fit_tensor_spline, traced_spline_contour
W='/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s19-codex-20260916/fsa-coil-edit-panel-lcfs-2'
z=np.load(W+'/docs/figures/forward-solve-api/coil-edit-nonconvergence/panel-states.npz',allow_pickle=False)
r=jnp.asarray(z['radius']); h=jnp.asarray(z['height'])
def show(i, dlevel=0.0):
    psi=jnp.asarray(z[f'psi_{i}']); xp=jnp.asarray(z[f'xpoints_{i}']).reshape(-1 if 0 else -1,2)
    sp=fit_tensor_spline(r,h,psi); lvl=float(sp(xp[0,0],xp[0,1]))+dlevel
    c=traced_spline_contour(psi,r,h,jnp.asarray(lvl))
    ep=np.asarray(c['segment_endpoints_rz']).reshape(-1,2,2)
    v=np.asarray(c['segment_valid']).reshape(-1).astype(bool)
    ctr=np.asarray(c['segment_controls_rz']).reshape(-1,4,2)
    seg=ep[v]
    # unique nodes by coordinate hashing
    pts=np.round(seg.reshape(-1,2),9)
    key={}
    ids=np.zeros(len(pts),dtype=int)
    for k,p in enumerate(pts):
        t=tuple(p)
        if t not in key: key[t]=len(key)
        ids[k]=key[t]
    ids=ids.reshape(-1,2)
    deg=np.zeros(len(key),dtype=int)
    for a,b in ids: deg[a]+=1; deg[b]+=1
    print(f'--- state {i} dlevel={dlevel} lvl={lvl:.6f} segments={v.sum()} nodes={len(key)}')
    # connected components over segments
    parent=list(range(len(key)))
    def find(a):
        while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
        return a
    for a,b in ids:
        ra,rb=find(a),find(b)
        if ra!=rb: parent[ra]=rb
    from collections import defaultdict
    groups=defaultdict(list)
    for n in range(len(key)): groups[find(n)].append(n)
    inv={vv:kk for kk,vv in key.items()}
    for root,members in sorted(groups.items(), key=lambda kv:-len(kv[1])):
        ends=[m for m in members if deg[m]==1]
        loop = len(open := [e for e in ends])
        print(f'   comp size={len(members)} degree1={len(ends)}')
        for m in ends:
            rr,zz=inv[m]
            onb = abs(rr-0.06)<1e-6 or abs(rr-2.0)<1e-6 or abs(zz+2.0)<1e-6 or abs(zz-2.0)<1e-6
            print(f'      end r={rr:.4f} z={zz:.4f} on_box_boundary={onb}')
for d in (0.0,):
    show(0,d); show(13,d)
