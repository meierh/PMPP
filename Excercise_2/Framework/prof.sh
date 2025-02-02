#!/bin/bash

mkdir -p out
rm -f out/*

~/.nsight/ncu -o out/ex2_coalescing --import-source on -f build/pmpp_ex2_sol -t1;
~/.nsight/ncu -o out/ex2_conflicts --import-source on -f build/pmpp_ex2_sol -t2;
~/.nsight/ncu -o out/ex2_roofline --import-source on -f build/pmpp_ex2_sol -t3;
~/.nsight/ncu -o out/ex2_divergence --import-source on -f build/pmpp_ex2_sol -t4

tar -czvf out.tar.gz out
