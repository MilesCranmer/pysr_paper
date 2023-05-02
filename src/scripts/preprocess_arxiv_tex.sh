#!/bin/bash
pushd $1
rm **/*.bib
sed -i ms.tex -e 's/finalizecache/frozencache/g'
sed -i ms.tex -e '/PassOptionsToPackage/d'
sed -i showyourwork.tex -e 's/RequirePackage{xcolor}/RequirePackage[table]{xcolor}/g'
popd
