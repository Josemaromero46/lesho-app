# Latexmk configuration for XeLaTeX with Biber and Glossaries
$pdf_mode = 5;  # Use xelatex and xdvipdfmx
$postscript_mode = $dvi_mode = 0;

# Set XeLaTeX as the compiler
$xelatex = 'xelatex %O -shell-escape -interaction=nonstopmode %S';

# Use biber for bibliography
$biber = 'biber %O %S';
$bibtex_use = 2;

# Add glossaries support
add_cus_dep('glo', 'gls', 0, 'run_makeglossaries');
add_cus_dep('acn', 'acr', 0, 'run_makeglossaries');
add_cus_dep('slo', 'sls', 0, 'run_makeglossaries');

sub run_makeglossaries {
    my ($base_name, $path) = fileparse($_[0]);
    pushd $path;
    my $return = system "makeglossaries", $base_name;
    popd;
    return $return;
}

# Clean up extra files
$clean_ext = "acn acr alg aux bbl bcf blg fls glo gls glg ist lof log lot out run.xml synctex.gz toc xdv fdb_latexmk mtc mtc0 mtc1 mtc2 mtc3 maf";
