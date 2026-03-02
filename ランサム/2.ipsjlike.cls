%% ipsjlike.cls — faithful approximation of IPSJ Transactions format
%% Based on IPSJ Journal (情報処理学会論文誌) style guidelines
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ipsjlike}[2024/01/01 IPSJ-like journal class]

\LoadClass[10pt,a4paper,twocolumn]{article}

%% ---- geometry ----
\RequirePackage[a4paper,
  top=24mm, bottom=22mm,
  left=17mm, right=17mm,
  columnsep=6mm,
  headheight=14pt,
  headsep=4mm]{geometry}

%% ---- fonts & language ----
\RequirePackage{fontspec}
\setmainfont{Noto Serif CJK JP}
\setsansfont{Noto Sans CJK JP}
\setmonofont{DejaVu Sans Mono}[Scale=0.82]

%% ---- math ----
\RequirePackage{amsmath,amssymb,amsthm,mathtools,bm}

%% ---- tables / figures ----
\RequirePackage{booktabs,tabularx,array,multirow}
\RequirePackage{graphicx,float}

%% ---- color ----
\RequirePackage[dvipsnames,table]{xcolor}
\definecolor{IPSJblue}{RGB}{0,60,120}
\definecolor{IPSJgray}{RGB}{90,90,90}
\definecolor{IPSJlightgray}{RGB}{240,240,245}
\definecolor{IPSJrule}{RGB}{0,60,120}
\definecolor{shadecolor}{RGB}{245,245,250}

%% ---- boxes ----
\RequirePackage{tcolorbox}
\tcbuselibrary{skins,breakable}

%% ---- misc ----
\RequirePackage{fancyhdr,titlesec,microtype,parskip,enumitem}
\RequirePackage{cite}
\RequirePackage[colorlinks=true,linkcolor=IPSJblue,citecolor=IPSJblue,urlcolor=IPSJblue,
                breaklinks=true]{hyperref}

%% ---- headers ----
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\fontsize{7}{8}\selectfont\color{IPSJgray}情報処理学会論文誌\ Vol.~XX\ No.~X\ 1--18 (XXXX)}
\fancyhead[R]{\fontsize{7}{8}\selectfont\color{IPSJgray}\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}

%% ---- section titles ----
\titleformat{\section}
  {\normalsize\bfseries\color{IPSJblue}}{\thesection.}{0.4em}{}
\titleformat{\subsection}
  {\small\bfseries\color{IPSJblue}}{\thesubsection.}{0.4em}{}
\titleformat{\subsubsection}
  {\small\bfseries}{\thesubsubsection.}{0.4em}{}
\titlespacing*{\section}{0pt}{8pt}{3pt}
\titlespacing*{\subsection}{0pt}{5pt}{2pt}
\titlespacing*{\subsubsection}{0pt}{4pt}{1pt}

%% ---- IPSJ title block (full width via \twocolumn[...]) ----
\newcommand{\IPSJtitleblock}[1]{%
  \twocolumn[{%
    \noindent\rule{\linewidth}{1.2pt}\vspace{1mm}\\
    #1
    \vspace{1mm}\noindent\rule{\linewidth}{0.6pt}
    \vspace{4mm}
  }]%
}

%% ---- theorem styles ----
\theoremstyle{plain}
\newtheorem{theorem}{定理}[section]
\newtheorem{lemma}[theorem]{補題}
\newtheorem{proposition}[theorem]{命題}
\newtheorem{corollary}[theorem]{系}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{定義}
\newtheorem{assumption}[theorem]{仮定}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{注意}

%% ---- abstract boxes ----
\newenvironment{jabstract}{%
  \begin{tcolorbox}[enhanced,
    colback=IPSJlightgray, colframe=IPSJblue,
    boxrule=0.5pt, left=5pt, right=5pt, top=4pt, bottom=4pt,
    title={\fontsize{8}{10}\bfseries\color{white}あらまし},
    attach boxed title to top left={yshift=-2mm,xshift=4mm},
    boxed title style={colback=IPSJblue,sharp corners},
    breakable]
  \fontsize{8.5}{12}\selectfont
}{%
  \end{tcolorbox}
}

\newenvironment{enabstract}{%
  \begin{tcolorbox}[enhanced,
    colback=IPSJlightgray, colframe=IPSJgray,
    boxrule=0.5pt, left=5pt, right=5pt, top=4pt, bottom=4pt,
    title={\fontsize{8}{10}\bfseries\color{white}Abstract},
    attach boxed title to top left={yshift=-2mm,xshift=4mm},
    boxed title style={colback=IPSJgray,sharp corners},
    breakable]
  \fontsize{8.5}{12}\selectfont\itshape
}{%
  \end{tcolorbox}
}

%% ---- row colors ----
\definecolor{rowA}{RGB}{245,249,252}

%% ---- proof box ----
\tcbset{
  proofbox/.style={
    enhanced, breakable,
    colback=white, colframe=IPSJgray!40,
    borderline west={2pt}{0pt}{IPSJblue},
    boxrule=0pt, left=8pt, right=4pt, top=3pt, bottom=3pt,
    fonttitle=\small\bfseries,
  },
  remarkbox/.style={
    enhanced, breakable,
    colback=IPSJlightgray, colframe=IPSJgray!50,
    boxrule=0.4pt, left=6pt, right=6pt, top=3pt, bottom=3pt,
    fonttitle=\small\bfseries,
  },
}

\endinput
