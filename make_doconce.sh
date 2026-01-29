#!/bin/bash
set -e

# Base filename without extension
name=preliminaries
#name=index
# name=interactive_gstar_function_v2
#name=monte_carlo_v2
#name=introduction_gpc_v2


function generate_ipynb {
    name=$1
#    local local_ipynb_options='--encoding=utf-8 --ipynb_admon=hrule --ipynb_disable_mpl_inline'
#    echo "local variable $local_ipynb_options"
#   doconce format ipynb $name $local_ipybn_options
#    doconce format ipynb $name --encoding=utf-8 --ipynb_admon=hrule --ipynb_disable_mpl_inline --ipynb_cite=latex-plain
    doconce format ipynb $name --encoding=utf-8 --ipynb_admon=hrule --ipynb_disable_mpl_inline --ipynb_cite=latex-plain 
}


# Format to HTML
#doconce format html $name --html_style=solarized

generate_ipynb $name

