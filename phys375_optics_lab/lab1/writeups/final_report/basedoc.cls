%% DOC.CLS
%%
%%  Document Class
%%
%% All options are passed to the underlying article class implementation
%%
%% The following macros are defined:
%%
%%   \title{title}
%%      document title
%%   \author{authors}
%%      list of authors
%%   \institution{institution}
%%      list of author's institutions
%%   \date{date}
%%      publication date
%%   \dccnumber{T}{YY}{NNNN}{VV}{G}
%%       document control number
%%        T     document type identifier
%%        YY    two digit year
%%        NNNN  four digit serial number
%%        VV    two digit version number
%%        G     working group identifier
%%   \distribution{group}
%%      includes ``Distribution of this document: <group>'' on the title page
%%   \note{note}
%%      includes the specified note on the title page
%%   \internal
%%      sets note to ``This is an internal working note of the  project''
%%   \published{journal}
%%      sets note to ``To be published in <journal>''
%%   \submitted{journal}
%%      sets note to ``Submitted for publication in <journal>''
%%   \draft
%%      includes the label ``Draft'' on every page
%%   \restricted{group}
%%      includes the label ``Circulation resitricted to <group>'' on every page
%%   \sites
%%      include  site information on the title page
%%   \cvsid{$Id: doc.cls,v 1.1 2008/02/24 07:11:08 rana Exp $}
%%      include the file's CVS ID on the title page
%%
%% Shourov K. Chatterji
%% shourov@.mit.edu
%% 2003-Jun-18
%%
%% edited by Rana, July 2006 - 2008

%---------- LATEX VERSION ----------
\NeedsTeXFormat{LaTeX2e}

%---------- CLASS NAME AND REVISION ----------
\ProvidesClass{doc}[2006/07/17  document class]

%---------- PROCESS OPTIONS ----------
\DeclareOption{letterpaper}{\PassOptionsToClass{\CurrentOption}{article}}
\DeclareOption{12pt}{\PassOptionsToClass{\CurrentOption}{article}}
\DeclareOption{oneside}{\PassOptionsToClass{\CurrentOption}{article}}
\DeclareOption{onecolumn}{\PassOptionsToClass{\CurrentOption}{article}}
\DeclareOption{final}{\PassOptionsToClass{\CurrentOption}{article}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ExecuteOptions{letterpaper,12pt,oneside,onecolumn,final}
\ProcessOptions\relax

%---------- LOAD PARENT CLASS ----------
\LoadClass{article}

%---------- LOAD REQUIRED PACKAGES ----------
\RequirePackage{latexsym}
\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{amssymb}
\RequirePackage{graphicx}
\RequirePackage[usenames,dvipsnames]{color}
\RequirePackage{hyperref}
\RequirePackage{ifthen}
\RequirePackage{array}
\RequirePackage{calc}
\RequirePackage{vmargin}

%---------- SET MARGINS, INDENTATION, AND SPACING ----------
\setpapersize[portrait]{USletter}
\setmarginsrb{1in}{0.5in}{1in}{0.5in}{0.3in}{0.2in}{0.3pt}{0.2in}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5\baselineskip}

%---------- SET PAGE COLOR ----------
\pagecolor{white}

%---------- DEFAULT  DOCUMENT PROPERTIES ----------
\def\@dcctype{\relax}
\def\@dccyear{\relax}
\def\@dccserial{\relax}
\def\@dccversion{\relax}
\def\@dccgroup{\relax}
\def\@dccnumber{\relax}
\def\@distribution{\relax}
\def\@draft{\relax}
\def\@restricted{\relax}
\def\@note{\relax}
\def\@sites{\relax}
\def\@institution{\relax}
\def\@cvsid{\relax}

%---------- MACRO TO PARSE CVS ID TAG ----------
\def\setcvsid#1{\def\next##1#1{\def\@cvsid{\mbox{%
  CVS ##1 --- Processed with \LaTeX\ on \today}}}\next}

%---------- COMMANDS TO SET  DOCUMENT PROPERTIES ----------
\newcommand{\dccnumber}[5]{%
  \def\@dcctype{#1}%
  \def\@dccyear{#2}%
  \def\@dccserial{#3}%
  \def\@dccversion{#4}%
  \def\@dccgroup{#5}%
  \def\@dccnumber{-#1#2#3-#4-#5}%
  \ifthenelse{\equal\@dcctype{C}}{%
    \def\@doctype{Contractual Note}%
  }{%
  \ifthenelse{\equal\@dcctype{D}}{%
    \def\@doctype{Drawing}%
  }{%
  \ifthenelse{\equal\@dcctype{E}}{%
    \def\@doctype{Engineering Note}%
  }{%
  \ifthenelse{\equal\@dcctype{G}}{%
    \def\@doctype{Presentation}%
  }{%
  \ifthenelse{\equal\@dcctype{L}}{%
    \def\@doctype{Correspondence}%
  }{%
  \ifthenelse{\equal\@dcctype{M}}{%
    \def\@doctype{Administrative Note}%
  }{%
  \ifthenelse{\equal\@dcctype{P}}{%
    \def\@doctype{Publication}%
  }{%
  \ifthenelse{\equal\@dcctype{T}}{%
    \def\@doctype{Technical Note}%
  }{%
    \def\@doctype{}%
  }}}}}}}}%
}
\newcommand{\draft}{\def\@draft{Draft}}
\newcommand{\restricted}[1]{\def\@restricted{%
  Circulation restricted to #1}}
\newcommand{\distribution}[1]{\def\@distribution{%
  \textit{Distribution of this document:}\vspace{1ex}\\#1}}
\newcommand{\note}[1]{\def\@note{#1}}
\newcommand{\internal}{\def\@note{%
  This is an internal working\\note of the  project}}
\newcommand{\published}[1]{\def\@note{%
  To be published in\\#1}}
\newcommand{\submitted}[1]{\def\@note{%
  Submitted for publication in\\#1}}
\newcommand{\sites}{\def\@sites{true}}
\newcommand{\institution}[1]{\def\@institution{\\#1}}
\newcommand{\cvsid}[1]{\setcvsid#1}

% Colorized Section,subsSection,& subsub
\renewcommand\section{\@startsection {section}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {1.4ex \@plus.2ex}%
                                   {\color{red}\normalfont\Large\bfseries}}
\renewcommand{\subsection}{\@startsection{subsection}{2}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {0.5ex \@plus .2ex}%
                                     {\color{blue}\normalfont\bfseries}}
\renewcommand{\subsubsection}{\@startsection{subsubsection}{3}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {0.2ex \@plus .2ex}%
                                     {\color[named]{OliveGreen}\normalfont\bfseries}}


%---------- REDEFINE DATE FORMAT ----------
\renewcommand{\today}{%
  \number\year/\ifnum\month<10 0\fi\number\month/\ifnum\day<10 0\fi\number\day}

%---------- ? ----------
\newcommand{\PreserveBackslash}[1]{\let\temp=\\#1\let\\=\temp}

%---------- LABEL LAST PAGE ----------
\AtEndDocument{\label{doc-lastpage}}

%---------- DEFINE FOOTERS AND HEADERS ----------
\def\ps@headings{
  \def\@evenfoot{\vbox to 0in{%
     \centering\mbox{page \thepage{}}%
     }}
  \def\@oddfoot{\vbox to 0in{%
     \centering\mbox{page \thepage{}}%
     }}
  \def\@oddhead{\vbox to 0in{%
     \vskip -0.3in%
     \centering\mbox{\@dccnumber}\rule{0pt}{0ex}\\
     \vspace{0.1in}
     \centering\mbox{\color{red} \@draft%
     \ifthenelse{\equal\@draft{\relax}}{}{%
       \ifthenelse{\equal\@restricted{\relax}}{}{:\ }}%
     \@restricted}%
     }}
  \def\@evenhead{\vbox to 0in{%
     \vskip -0.3in%
     \centering\mbox{\@dccnumber}\rule{0pt}{0ex}\\
     \vspace{0.1in}
     \centering\mbox{\color{red} \@draft%
     \ifthenelse{\equal\@draft{\relax}}{}{%
       \ifthenelse{\equal\@restricted{\relax}}{}{:\ }}%
     \@restricted}%
     }}
}

%---------- ENABLE HEADERS AND FOOTERS ----------
\pagestyle{headings}

%---------- DEFINE TITLEPAGE ----------
\renewcommand\maketitle{%
  \thispagestyle{titleheadings}%
  \begin{center}
    \fontsize{12pt}{14pt}\selectfont%
    \begin{tabular*}{5.5in}%
      {|*{1}{>{\PreserveBackslash\centering\hspace{0pt}}%
             p{1.5in-\tabcolsep*2-\arrayrulewidth*2/3}}%
        *{1}{>{\PreserveBackslash\centering\hspace{0pt}}%
             p{2.5in-\tabcolsep*2-\arrayrulewidth*2/3}}%
        *{1}{>{\PreserveBackslash\centering\hspace{0pt}}%
             p{1.5in-\tabcolsep*2-\arrayrulewidth*2/3}}|}
      \hline
      \textbf{\@doctype} &
      \textbf{\@dccnumber} &
      \@date\rule[-2ex]{0pt}{5ex}\\
      \hline
      \multicolumn{3}{|p{5.5in-\tabcolsep*2-\arrayrulewidth*2}|}%
        {\rule{0pt}{5ex}\centering\parbox[t]{5in}%
          {\centering\fontsize{22pt}{26pt}\selectfont%
           \bfseries\@title\rule[-2ex]{0pt}{0pt}}}\\
      \hline
      \multicolumn{3}{|p{5.5in-\tabcolsep*2-\arrayrulewidth*2}|}%
        {\rule{0pt}{3ex}\centering\@author\textit{\@institution}%
         \rule[-2ex]{0pt}{0ex}}\\
      \hline
    \end{tabular*}\\

    \vspace{0.5in}
    \@distribution\rule{0pt}{0ex}\\
    \vspace{0.25in}
    {\color{red}\fontsize{14pt}{14pt}\selectfont\@draft\rule{0pt}{0ex}}\\
    {\color{red}\fontsize{14pt}{14pt}\selectfont\@restricted\rule{0pt}{0ex}}\\
    \vspace{0.25in}
    \@note\rule{0pt}{0ex}\\
    \vfill
    \begin{tabular}{c@{\qquad}c}
      \textbf{University of Maryland} \\
      \textbf{Physics Department} \\
      \textbf{College Park, MD 20742} \\
      \textbf{Physics 375 - Experimental Physics III:} \\ \textbf{Electromagnetic Waves, Optics and Modern Physics} \\
    \end{tabular}\\[1ex]
  \end{center}
  \normalsize
  \newpage
}
\def\ps@titleheadings{%
  \let\@evenhead\@empty\let\@oddhead\@empty
  \def\@oddfoot{\hfil\mbox{\fontsize{9pt}{11pt}\selectfont\@cvsid}\hfil}%
  \def\@evenfoot{\hfil\mbox{\fontsize{9pt}{11pt}\selectfont\@cvsid}\hfil}%
}

%---------- AUTOMATICALLY GENERATE TITLE PAGE ----------
\AtBeginDocument{%
  \maketitle
  \setcounter{page}{1}
}
