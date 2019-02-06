#!/bin/bash

set -eux
export syn=true
export hyp=true
export mer=true

export mdl="syn_hyp_mer"

. scripts/wikitext103/wikitext103_base.sh
