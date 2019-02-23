#!/bin/bash

set -eux
export hyp=true
export data="wikitext2"
. scripts/run_once.sh
