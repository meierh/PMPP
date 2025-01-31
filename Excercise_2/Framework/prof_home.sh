#!/bin/bash

mkdir -p out
rm -f out/*

sudo ~/.nsight/ncu -o out/ex2_coalescing  --section SpeedOfLight --metrics smsp__sass_branch_targets_threads_divergent.avg,smsp__sass_branch_targets_threads_divergent.sum,smsp__pcsamp_sample_count,smsp__pcsamp_warps_issue_stalled_barrier,smsp__pcsamp_warps_issue_stalled_branch_resolving  --import-source on -f build/pmpp_ex2_sol -t1

sudo ~/.nsight/ncu -o out/ex2_conflicts --section MemoryWorkloadAnalysis --metrics smsp__pcsamp_warps_issue_stalled_lg_throttle,smsp__pcsamp_warps_issue_stalled_long_scoreboard,group:memory__shared_table --import-source on -f build/pmpp_ex2_sol -t2

sudo ~/.nsight/ncu -o out/ex2_roofline  --section SpeedOfLight_RooflineChart --metrics smsp__sass_branch_targets_threads_divergent.avg,smsp__sass_branch_targets_threads_divergent.sum,smsp__pcsamp_sample_count,smsp__pcsamp_warps_issue_stalled_barrier,smsp__pcsamp_warps_issue_stalled_branch_resolving --import-source on -f build/pmpp_ex2_sol -t3

sudo ~/.nsight/ncu -o out/ex2_divergence --section LaunchStats --metrics smsp__sass_branch_targets_threads_divergent.avg,smsp__sass_branch_targets_threads_divergent.sum,smsp__pcsamp_sample_count,smsp__pcsamp_warps_issue_stalled_barrier,smsp__pcsamp_warps_issue_stalled_branch_resolving --import-source on -f build/pmpp_ex2_sol -t4

tar -czvf out.tar.gz out
