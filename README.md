# Inferência bayesiana com dados H(z)

Repositório didático para ajustar modelos cosmológicos planos **LCDM** e **wCDM** a medidas observacionais do parâmetro de Hubble, **H(z)**, obtidas por cronômetros cósmicos.

O projeto segue verossimilhança gaussiana, estatística chi-quadrado, priors uniformes, MCMC com `emcee`, inicialização por Evolução Diferencial, AIC, BIC e evidência bayesiana aproximada por integração termodinâmica.


## Dados

O arquivo `Hz.csv` contém três colunas:

| coluna | significado |
|---|---|
| `z` | redshift |
| `H(z)` | parâmetro de Hubble observado, em km s^-1 Mpc^-1 |
| `errH` | incerteza 1-sigma de H(z) |

## Modelos implementados

### LCDM plano

```text
H(z) = H0 * sqrt[Omega_m * (1 + z)^3 + (1 - Omega_m)]
H0 = 100 h
```

Parâmetros livres:

```text
Omega_m, h
```

### wCDM plano

```text
H(z) = H0 * sqrt[Omega_m * (1 + z)^3 + (1 - Omega_m) * (1 + z)^(3(1+w))]
H0 = 100 h
```

Parâmetros livres:

```text
Omega_m, h, w
```

Quando `w = -1`, o modelo wCDM volta ao caso LCDM.

## Instalação

Crie um ambiente virtual e instale as dependências:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

No Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Como rodar

Execução padrão:

```bash
python CosmoFit.py --data Hz.csv --out results
```

Execução rápida para testar se tudo está funcionando:

```bash
python CosmoFit.py --data Hz.csv --out results --steps-post 2000 --steps-ti 500 --prior-samples 5000
```

## Saídas geradas

Para cada modelo, o script salva:

```text
results/LCDM_flat_posterior_samples.txt
results/LCDM_flat_corner.png
results/LCDM_flat_bestfit.png
results/LCDM_flat_ti_curve.txt
results/wCDM_flat_posterior_samples.txt
results/wCDM_flat_corner.png
results/wCDM_flat_bestfit.png
results/wCDM_flat_ti_curve.txt
results/summary.md
```

O arquivo `summary.md` resume medianas, intervalos de credibilidade, AIC, BIC e log-evidência.

## Observações importantes

- Os limites de prior estão definidos dentro de `default_models()` no script.
- A integração termodinâmica é uma aproximação numérica; para resultados finais, aumente `--steps-ti`, `--n-betas` e `--prior-samples`.
- Para publicação, rode cadeias mais longas e verifique convergência, autocorrelação e estabilidade da evidência.
- Os resultados podem variar levemente com a semente aleatória e com o tamanho das cadeias.

## Licença sugerida

Use uma licença aberta, como MIT, caso queira publicar no GitHub.
