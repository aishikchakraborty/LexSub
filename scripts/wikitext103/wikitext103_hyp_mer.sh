#!/bin/bash

set -eux
export mer=true
export hyp=true
export data="wikitext103"
. scripts/run_once.sh
