"""Numerical checks for client-side fixed-map insertion, without a browser runtime."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[2] / "mini_trainer/visualization/prototype_space/insertion.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is required for the browser numerical implementation")
def test_fixed_map_insertion_and_gradient():
    runner = r"""
const {insertionAffinities,insertionObjective,insertIntoFixedTSNE}=require(process.argv[1]);
const points=Array.from({length:100},(_,i)=>[10*Math.cos(i*2*Math.PI/100),10*Math.sin(i*2*Math.PI/100)]);
const prototypes=Float32Array.from(points.flatMap(([x,y])=>[x/10,y/10,0]));
const before=JSON.stringify(points),aff=insertionAffinities(points.map(([x,y])=>Math.hypot(x-8,y-2)),10);
const q=[8.2,2.3],state=insertionObjective(q,points,aff),epsilon=1e-5;
const finite=q.map((_,j)=>{const a=q.slice(),b=q.slice();a[j]+=epsilon;b[j]-=epsilon;
return (insertionObjective(a,points,aff).loss-insertionObjective(b,points,aff).loss)/(2*epsilon);});
const result=insertIntoFixedTSNE([1,.01,0],prototypes,points,{perplexity:5});
console.log(JSON.stringify({gradient:state.gradient,finite,result,unchanged:before===JSON.stringify(points)}));
"""
    output = subprocess.run(["node", "-e", runner, str(SCRIPT)], capture_output=True, text=True, check=True)
    report = json.loads(output.stdout)
    np.testing.assert_allclose(report["gradient"], report["finite"], rtol=1e-6, atol=1e-8)
    assert report["unchanged"]
    assert report["result"]["kl"] <= report["result"]["initial_kl"]
    assert report["result"]["retained_fraction"] >= 0.9
    assert np.isfinite(report["result"]["coordinates"]).all()
