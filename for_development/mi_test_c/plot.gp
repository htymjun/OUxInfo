# 出力ファイル名と形式を設定
set terminal pngcairo size 800,800 enhanced font 'Verdana,12'
set output 'mi_graph.png'

# グラフの各種設定
set title "Mutual Information vs. Variance of Y"
set xlabel "Var(Y)" font ",14"
set ylabel "Mutual Information" font ",14"
set grid

# 凡例の設定
set key top right

# CSVファイルの区切り文字を設定
set datafile separator ","

# データをプロット
plot 'results.csv' using 1:2 with lines lw 2 lc 'black' title 'Theoretical', \
     '' using 1:3 with points pt 7 ps 1.5 lc 'blue' title 'k=5', \
     '' using 1:4 with points pt 7 ps 1.5 lc 'red' title 'k=10', \
     '' using 1:5 with points pt 7 ps 1.5 lc 'dark-green' title 'k=50'