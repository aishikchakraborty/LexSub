#!/bin/bash

set -eux
export syn=true
export mer=true
export data="glove"
export mdl="retro"

. scripts/wikitext2/wikitext2_base.sh
