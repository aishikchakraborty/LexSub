#!/bin/bash

set -eux
export syn=true
export hyp=true
export mdl="syn_hyp"

. scripts/wikitext103/wikitext103_base.sh
