# TOP Secrety
Go away or be here and discrete.

## Gauss-Newton — vizualizácia (single shooting)

```bash
pip install -r requirements.txt
python gn_vizualizacia.py                    # y' = αy + βt,   p* = (3, 9)
python gn_vizualizacia.py --model bernoulli  # y' = αy + βty², p* = (1, 2), T = 0.8·t*
python gn_vizualizacia.py --p0 0.5 20 --sigma 2 --n 200
```

| Súbor | Obsah |
|---|---|
| `gauss_newton.py` | GN (čistý / s polením kroku), Jakobián cez rovnice citlivosti, história iterácií |
| `modely.py` | analytické a RK45 riešenia modelov z DP |
| `viz.py` | kánon grafov + `gn_cesta_na_konturach`, `gn_konvergencia`, `gn_fit_iteracie` |
| `gn_vizualizacia.py` | generuje grafy do `vysledky_grafy/` |
