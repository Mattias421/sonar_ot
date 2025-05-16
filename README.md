```
pip install fairseq2 --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.6.0/cu124
```
```
pip install sonar-space
```

```
awk 'BEGIN{OFS="\t"} NR==1{print "ID", $0; next} {print NR-1, $0}' input.tsv > output.tsv
```

`FAIRSEQ2_USER_ASSET_DIR`
