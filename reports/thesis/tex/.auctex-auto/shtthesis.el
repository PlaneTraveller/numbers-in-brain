(TeX-add-style-hook
 "shtthesis"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("xcolor" "hyperref" "table") ("ulem" "normalem") ("enumitem" "shortlabels" "inline") ("unicode-math" "mathbf=sym")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "iftex"
    "kvdefinekeys"
    "kvsetkeys"
    "kvoptions"
    "datetime"
    "ctexbook"
    "ctexbook10"
    "expl3"
    "xparse"
    "xcolor"
    "geometry"
    "calc"
    "verbatim"
    "etoolbox"
    "ifthen"
    "graphicx"
    "indentfirst"
    "ulem"
    "fancyhdr"
    "lastpage"
    "tocvsec2"
    "letltxmacro"
    "fontspec"
    "caption"
    "enumitem"
    "mathtools"
    "amsthm"
    "unicode-math"
    "biblatex"
    "hyperref")
   (TeX-add-symbols
    '("header" ["argument"] 2)
    '("item" ["argument"] 2)
    '("intobmkstar" ["argument"] 2)
    '("intobmknostar" ["argument"] 3)
    '("intotocstar" ["argument"] 2)
    '("intotocnostar" 3)
    '("eqref" 1)
    '("chaptermark" 1)
    "version"
    "versiondate"
    "ShtThesis"
    "sht"
    "shtsetup"
    "frontmatter"
    "mainmatter"
    "currentfontset"
    "songti"
    "heiti"
    "kaishu"
    "fangsong"
    "bm"
    "square"
    "artxmaincnt"
    "intotoc"
    "intobmk"
    "makeindices"
    "textbf"
    "makedeclarations"
    "makebiblio"
    "backmatter"
    "contentsname"
    "listfigurename"
    "listtablename"
    "appendixname"
    "indexname"
    "refname"
    "bibname"
    "tablename"
    "figurename"
    "le"
    "ge"
    "artxaux"
    "ULthickness"
    "tabcolsep"
    "arraystretch"
    "clearpage")
   (LaTeX-add-environments
    '("nomenclatures" LaTeX-env-args ["argument"] 0)
    '("abstract*" LaTeX-env-args ["argument"] 0)
    '("abstract" LaTeX-env-args ["argument"] 0)
    "resume"
    "publications"
    "publications*"
    "patents"
    "patents*"
    "projects"
    "acknowledgement")
   (LaTeX-add-pagestyles
    "Plain"
    "RomanNumbered"
    "LRNumbered"
    "LRNumberedAppendix"
    "RomanNumberedWithLogo"
    "MNNumberedWithLogo")
   (LaTeX-add-lengths
    "sht")
   (LaTeX-add-xcolor-definecolors
    "fdu@link"
    "fdu@url"
    "fdu@cite"
    "ShtRed")
   (LaTeX-add-caption-DeclareCaptions
    '("\\DeclareCaptionFont{wuhaocuti}" "Font" "wuhaocuti"))
   (LaTeX-add-mathtools-newtagforms
    "dots")
   (LaTeX-add-amsthm-newtheorems
    "theorem"
    "lemma"
    "corollary"
    "proposition"
    "conjecture"
    "definition"
    "axiom"
    "example"
    "exercise"
    "problem"
    "remark"))
 :latex)

