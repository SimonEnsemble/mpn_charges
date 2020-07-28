#!/bin/bash
date
~/julia-1.4.1/bin/julia -p 1 run_henry.jl $xtal $method
