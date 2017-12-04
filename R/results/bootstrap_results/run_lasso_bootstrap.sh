#!/bin/bash

seed=$(($SLURM_ARRAY_TASK_ID))

Rscript ../../fit_lasso_bootstrap.R seed
