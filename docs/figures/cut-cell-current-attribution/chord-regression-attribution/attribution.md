<h3>Chord map-fidelity regression — revision attribution</h3>

Two chord rows of <code>benchmarks/plasma_cell_map_fidelity.py</code> at
requested &minus;110 cells (135 realised), analytic-state map, no solve, on the
CPU witness (<code>JAX_PLATFORMS=cpu), measured at six revisions. Each revision
was extracted with <code>git archive</code> under the run scratch and imported
by <code>PYTHONPATH</code> pointing at that extraction; each log header records
the revision, the extraction path and the imported
<code>nova.equilibrium.clip_quadrature.__file__</code>.

<h4>Method</h4>

<ul>
<li><b>de29a1a30, 6c519684e, 427f05e10, b0a0676df</b> — the benchmark at these
revisions asserts a GPU backend inline. Each row was run through that
revision's own <code>measure_pair</code> from a short driver that removes the
single backend-assertion line and nothing else (<code>run_rows.py</code>). Those
revisions predate the benchmark's centroid helper, so the published centroid
formula was reproduced in the driver and applied to the operator and moments the
row itself used.</li>
<li><b>fc13e55fb, 92613fd3d</b> — the benchmark carries
<code>--cpu-preflight</code>, so each row was measured through that flag
directly.</li>
</ul>

<h4>Rows</h4>

<table>
<tr><th>revision</th><th>case (chord, requested &minus;110)</th><th>sup_relative</th><th>rms_relative</th><th>centroid dR [mm]</th><th>centroid dZ [mm]</th></tr>
<tr><td>de29a1a30</td><td>weak-rotation-reactor-static</td><td>9.54029e-05</td><td>1.22702e-04</td><td>-0.06982</td><td>-7.179e-10</td></tr>
<tr><td>de29a1a30</td><td>diverted-single-null</td><td>0.125190</td><td>0.103865</td><td>-1.08960</td><td>-6.962e+00</td></tr>
<tr><td>6c519684e</td><td>weak-rotation-reactor-static</td><td>9.54029e-05</td><td>1.22702e-04</td><td>-0.06982</td><td>-7.179e-10</td></tr>
<tr><td>6c519684e</td><td>diverted-single-null</td><td>0.125190</td><td>0.103865</td><td>-1.08960</td><td>-6.962e+00</td></tr>
<tr><td>427f05e10</td><td>weak-rotation-reactor-static</td><td>9.54029e-05</td><td>1.22702e-04</td><td>-0.06982</td><td>-7.179e-10</td></tr>
<tr><td>427f05e10</td><td>diverted-single-null</td><td>0.125190</td><td>0.103865</td><td>-1.08960</td><td>-6.962e+00</td></tr>
<tr><td>fc13e55fb</td><td>weak-rotation-reactor-static</td><td>9.54029e-05</td><td>1.22702e-04</td><td>-0.06982</td><td>-7.179e-10</td></tr>
<tr><td>fc13e55fb</td><td>diverted-single-null</td><td>0.125190</td><td>0.103865</td><td>-1.08960</td><td>-6.962e+00</td></tr>
<tr><td>b0a0676df</td><td>weak-rotation-reactor-static</td><td>1.86063e-02</td><td>1.94080e-02</td><td>-0.64553</td><td>+7.671e-07</td></tr>
<tr><td>b0a0676df</td><td>diverted-single-null</td><td>0.277957</td><td>0.213516</td><td>-5.17730</td><td>+4.342e+00</td></tr>
<tr><td>92613fd3d</td><td>weak-rotation-reactor-static</td><td>1.86063e-02</td><td>1.94080e-02</td><td>-0.64553</td><td>+7.671e-07</td></tr>
<tr><td>92613fd3d</td><td>diverted-single-null</td><td>0.277957</td><td>0.213516</td><td>-5.17730</td><td>+4.342e+00</td></tr>
</table>

<p>Each raw row is the benchmark's own JSON receipt under <code>logs/&lt;revision&gt;/</code>;
<code>sup_relative</code> is <code>mismatch.sup_relative</code> and the centroid
is <code>support_current_centroid_offset_mm</code> (<code>dR</code>,
<code>dZ</code>).</p>

<h4>Attribution</h4>

<p><b>First revision where weak-rotation chord sup leaves &asymp;1e-4:</b>
<code>b0a0676df</code> — sup rises from 9.54029e-05 at <code>fc13e55fb</code>
to 1.86063e-02, a factor of 195, and the centroid offset rises from
&minus;0.070&nbsp;mm to &minus;0.646&nbsp;mm. The diverted chord row falls the
same way (0.125190 &rarr; 0.277957).</p>

<p><b>Last good revision:</b> <code>fc13e55fb</code> (<code>92613fd3d^1</code>)
— weak chord sup 9.54029e-05, identical to the committed receipt's revision
<code>de29a1a30</code> and to both sides of the exact-moment repair
(<code>427f05e10^1</code> = <code>6c519684e</code>, <code>427f05e10</code>), so
the exact-moment repair did not touch the chord row.</p>

<p><b>Commits between the last good and the first bad revision.</b>
<code>fc13e55fb</code> and <code>b0a0676df</code> share merge-base
<code>a71934910</code>; the commits reachable from <code>b0a0676df</code> but
not from <code>fc13e55fb</code> are exactly the chord-membership lineage:</p>

<ul>
<li><code>3ba7f6b00</code> test(equilibrium): pin exterior chord membership</li>
<li><code>a59529f20</code> fix(equilibrium): use connected separatrix membership
— the functional change</li>
<li><code>c8260b44c</code> docs(equilibrium): bank chord membership gate</li>
</ul>

<p>The functional change is <code>a59529f20</code>. It reaches the mainline at
the merge <code>92613fd3d</code> (parents <code>fc13e55fb</code> and
<code>cd8454dc8</code>), whose branch also carries <code>9acda6b6d</code> and
<code>cd8454dc8</code>. <code>b0a0676df</code> is the chord lineage after
<code>a59529f20</code> but before the applied-membership flag; it is bad there,
so the regression is carried by the chord-membership change itself, not by the
later flag or by the merge machinery.</p>