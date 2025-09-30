#!/bin/bash
# Translate .tex files in .. to doconce
set -x  # Show all commands prior to execution
#
function system {
  "$@"
  if [ $? -ne 0 ]; then
    echo "make.sh: unsuccessful command $@"
    echo "abort!"
    exit 1
  fi
}

#code=vrb
code=pyg

pdf_opt="-shell-escape"
options="--no_abort  --encoding=utf-8 --allow_refs_to_external_docs"

function compile {
name=$1
system doconce format pdflatex $name --latex_code_style=$code $options 
#system doconce format pdflatex $name --latex_code_style=$code --no_abort
system doconce replace 'subsection{Nomenclature}' 'centerline{\textbf{Nomenclature}}\par\vspace{4mm}\par' $name.tex
system doconce replace '10pt]{article}' '12pt, a4paper]{book}' $name.tex
system doconce replace '${bbox()}' '' $name.tex
system doconce replace '${ebox()}' '' $name.tex
#system doconce replace '\newcoomand' '\newcommand' newcommands.tex
#system doconce replace 'example_google' '"\texttt{example\_google}"' $name.tex
system pdflatex $pdf_opt $name
#system makeindex $name
system bibtex $name
system pdflatex $pdf_opt $name
system pdflatex $pdf_opt $name
}

function generate_html {
name=$1
doconce format html $name --encoding=utf-8 $doc_options  $exercise_options
doconce split_html $name.html --pagination --nav_button=top+bottom 
python preload_movie_html.py 
}



function generate_ipynb {
    name=$1
#    local local_ipynb_options='--encoding=utf-8 --ipynb_admon=hrule --ipynb_disable_mpl_inline'
#    echo "local variable $local_ipynb_options"
#   doconce format ipynb $name $local_ipybn_options
    doconce format ipynb $name --encoding=utf-8 --ipynb_admon=hrule --ipynb_disable_mpl_inline --ipynb_cite=latex-plain
}


exercise_options='--without_answers --without_solutions'
#ipynb_options='--encoding=utf-8 --ipynb_admon=hrule --ipynb_disable_mpl_inline' 
doc_options='--encoding=utf-8 --ipynb_admon=hrule $opt --without_solutions'

cd ./references
bash clean.sh
cd ..


# system doconce spellcheck -d ./.dict4spell.txt $name.do.txt
# Compile all doconce files
# for name in `ls *.do.txt`; do
#     generate_ipynb $name 
# done


#name=sensitivity_introduction_interactive.do.txt
#name=interactive_gstar_function.do.txt
name=test_doconce.do.txt
generate_ipynb $name 
