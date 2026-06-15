#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inferência bayesiana em cosmologia usando medidas H(z).

1. modelos cosmológicos planos LCDM e wCDM;
2. verossimilhança gaussiana para dados H(z) com erros independentes;
3. priors uniformes normalizados;
4. busca inicial do melhor ajuste por Evolução Diferencial;
5. amostragem da posterior com MCMC via emcee;
6. comparação de modelos por AIC, BIC e evidência bayesiana aproximada
   via integração termodinâmica.

Uso rápido:
    python CosmoFit.py --data data/Hz.csv --out results

Para um teste mais rápido, reduza --steps-post e --steps-ti.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution


# -----------------------------------------------------------------------------
# Estruturas de dados
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class HzData:
    """Guarda a tabela observacional de cronômetros cósmicos.

    z: redshift do ponto observado.
    H: valor observado de H(z), em km s^-1 Mpc^-1.
    sigma: incerteza 1-sigma associada a H(z).
    """

    z: np.ndarray
    H: np.ndarray
    sigma: np.ndarray


@dataclass(frozen=True)
class CosmologyModel:
    """Define um modelo cosmológico a ser ajustado aos dados."""

    name: str
    labels: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]
    H_of_z: Callable[[np.ndarray, np.ndarray], np.ndarray]

    @property
    def ndim(self) -> int:
        return len(self.bounds)


# -----------------------------------------------------------------------------
# Leitura dos dados
# -----------------------------------------------------------------------------

def load_hz_data(filename: str | Path) -> HzData:
    """Lê um CSV com colunas z, H(z) e errH.

    O arquivo incluído em data/Hz.csv usa aspas no cabeçalho; numpy ignora isso
    porque pulamos a primeira linha. O restante é numérico.
    """

    array = np.loadtxt(filename, delimiter=",", skiprows=1)
    return HzData(z=array[:, 0], H=array[:, 1], sigma=array[:, 2])


# -----------------------------------------------------------------------------
# Modelos cosmológicos do paper
# -----------------------------------------------------------------------------

def H_lcdm_flat(theta: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Modelo LCDM plano.

    Parâmetros:
        theta[0] = Omega_m, densidade de matéria hoje.
        theta[1] = h, com H0 = 100 h km s^-1 Mpc^-1.

    Equação:
        H(z) = H0 * sqrt[Omega_m (1+z)^3 + (1 - Omega_m)].
    """

    omega_m, h = theta
    H0 = 100.0 * h
    Ez2 = omega_m * (1.0 + z) ** 3 + (1.0 - omega_m)
    return H0 * np.sqrt(Ez2)


def H_wcdm_flat(theta: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Modelo wCDM plano com equação de estado constante w.

    Parâmetros:
        theta[0] = Omega_m.
        theta[1] = h, com H0 = 100 h.
        theta[2] = w, parâmetro da equação de estado da energia escura.

    Equação:
        H(z) = H0 * sqrt[Omega_m(1+z)^3
                         + (1-Omega_m)(1+z)^(3(1+w))].

    Quando w = -1, este modelo recupera o caso LCDM plano.
    """

    omega_m, h, w = theta
    H0 = 100.0 * h
    matter = omega_m * (1.0 + z) ** 3
    dark_energy = (1.0 - omega_m) * (1.0 + z) ** (3.0 * (1.0 + w))
    return H0 * np.sqrt(matter + dark_energy)


def default_models() -> list[CosmologyModel]:
    """Modelos e priors uniformes usados no exemplo."""

    return [
        CosmologyModel(
            name="LCDM_flat",
            labels=(r"$\Omega_m$", r"$h$"),
            bounds=((0.10, 0.50), (0.40, 0.90)),
            H_of_z=H_lcdm_flat,
        ),
        CosmologyModel(
            name="wCDM_flat",
            labels=(r"$\Omega_m$", r"$h$", r"$w$"),
            bounds=((0.10, 0.50), (0.40, 0.90), (-2.00, -0.30)),
            H_of_z=H_wcdm_flat,
        ),
    ]


# -----------------------------------------------------------------------------
# Estatística bayesiana: prior, chi2, log-verossimilhança e posterior
# -----------------------------------------------------------------------------

def chi2(theta: np.ndarray, model: CosmologyModel, data: HzData) -> float:
    """Calcula chi-quadrado para H(z).

    chi2 = sum_i [(H_obs_i - H_model_i) / sigma_i]^2.
    """

    prediction = model.H_of_z(theta, data.z)
    residual = (data.H - prediction) / data.sigma
    return float(np.sum(residual**2))


def log_likelihood(theta: np.ndarray, model: CosmologyModel, data: HzData) -> float:
    """Log-verossimilhança gaussiana completa.

    Mantemos o termo de normalização ln(2*pi*sigma^2). Ele não altera o melhor
    ajuste, mas é importante para comparar evidências bayesianas de forma
    consistente.
    """

    prediction = model.H_of_z(theta, data.z)
    residual = (data.H - prediction) / data.sigma
    return float(-0.5 * np.sum(residual**2 + np.log(2.0 * np.pi * data.sigma**2)))


def log_prior(theta: np.ndarray, bounds: Iterable[tuple[float, float]]) -> float:
    """Prior uniforme normalizado dentro dos limites físicos escolhidos.

    Fora dos limites, a probabilidade é zero e o log-prior é -infinito.
    Dentro dos limites, p(theta)=1/V, onde V é o volume do hipercubo de priors.
    """

    widths = []
    for value, (low, high) in zip(theta, bounds):
        if value <= low or value >= high:
            return -np.inf
        widths.append(high - low)
    return float(-np.sum(np.log(widths)))


def log_posterior(theta: np.ndarray, model: CosmologyModel, data: HzData) -> float:
    """Log-posterior usual: log prior + log verossimilhança."""

    lp = log_prior(theta, model.bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, model, data)


def log_posterior_tempered(
    theta: np.ndarray,
    model: CosmologyModel,
    data: HzData,
    beta: float,
) -> float:
    """Posterior temperada usada na integração termodinâmica.

    p_beta(theta|D) proporcional a prior(theta) * L(theta)^beta.
    beta=0 amostra o prior; beta=1 recupera a posterior comum.
    """

    lp = log_prior(theta, model.bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + beta * log_likelihood(theta, model, data)


# -----------------------------------------------------------------------------
# Otimização e MCMC
# -----------------------------------------------------------------------------

def find_map_with_de(model: CosmologyModel, data: HzData, seed: int) -> np.ndarray:
    """Encontra um ponto inicial de alta posterior por Evolução Diferencial.

    O MCMC funciona melhor quando os walkers começam próximos de uma região com
    boa probabilidade. A Evolução Diferencial é útil porque faz busca global sem
    precisar de gradientes.
    """

    def objective(theta: np.ndarray) -> float:
        lp = log_posterior(theta, model, data)
        return 1.0e100 if not np.isfinite(lp) else -lp

    result = differential_evolution(
        objective,
        bounds=model.bounds,
        seed=seed,
        polish=True,
        updating="immediate",
    )
    return np.asarray(result.x, dtype=float)


def initialize_walkers(
    theta0: np.ndarray,
    model: CosmologyModel,
    nwalkers: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Cria posições iniciais dos walkers perto do melhor ajuste.

    Se algum walker cair fora do prior, ele é reposicionado uniformemente dentro
    do intervalo permitido daquele parâmetro.
    """

    bounds = np.asarray(model.bounds, dtype=float)
    low = bounds[:, 0]
    high = bounds[:, 1]
    scale = 1.0e-2 * (high - low)
    p0 = theta0 + rng.normal(size=(nwalkers, model.ndim)) * scale

    for i in range(nwalkers):
        bad = (p0[i] <= low) | (p0[i] >= high)
        p0[i, bad] = low[bad] + rng.random(np.sum(bad)) * (high[bad] - low[bad])
    return p0


def run_mcmc(
    model: CosmologyModel,
    data: HzData,
    theta0: np.ndarray,
    nwalkers: int,
    nsteps: int,
    burn_frac: float,
    thin: int,
    seed: int,
    beta: float = 1.0,
) -> np.ndarray:
    """Executa emcee e devolve uma cadeia achatada após burn-in e thinning."""

    rng = np.random.default_rng(seed)
    p0 = initialize_walkers(theta0, model, nwalkers, rng)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        model.ndim,
        lambda th: log_posterior_tempered(th, model, data, beta),
    )
    sampler.run_mcmc(p0, nsteps, progress=True)

    burn = int(burn_frac * nsteps)
    return sampler.get_chain(discard=burn, thin=thin, flat=True)


# -----------------------------------------------------------------------------
# Critérios de comparação de modelos
# -----------------------------------------------------------------------------

def aic_bic(logL_max: float, k: int, n: int) -> tuple[float, float]:
    """Calcula AIC e BIC.

    AIC = 2k - 2 ln(Lmax)
    BIC = k ln(n) - 2 ln(Lmax)
    """

    return 2.0 * k - 2.0 * logL_max, k * np.log(float(n)) - 2.0 * logL_max


def thermodynamic_integration(
    model: CosmologyModel,
    data: HzData,
    theta0: np.ndarray,
    betas: np.ndarray,
    nwalkers: int,
    nsteps: int,
    burn_frac: float,
    thin: int,
    seed: int,
    n_prior_samples: int = 30_000,
) -> tuple[float, np.ndarray]:
    """Estima logZ = integral_0^1 <logL>_beta d beta.

    beta=0 é calculado por amostragem direta do prior uniforme.
    Para beta>0, fazemos MCMC da posterior temperada.
    """

    rng = np.random.default_rng(seed + 999)
    bounds = np.asarray(model.bounds, dtype=float)
    low, high = bounds[:, 0], bounds[:, 1]
    rows = []

    for beta in betas:
        beta = float(beta)
        if beta == 0.0:
            samples = low + rng.random((n_prior_samples, model.ndim)) * (high - low)
        else:
            samples = run_mcmc(
                model=model,
                data=data,
                theta0=theta0,
                nwalkers=nwalkers,
                nsteps=nsteps,
                burn_frac=burn_frac,
                thin=thin,
                seed=seed + int(1_000_000 * beta),
                beta=beta,
            )

        mean_logL = float(np.mean([log_likelihood(s, model, data) for s in samples]))
        rows.append((beta, mean_logL))

    curve = np.asarray(sorted(rows), dtype=float)
    logZ = float(np.trapz(curve[:, 1], curve[:, 0]))
    return logZ, curve


# -----------------------------------------------------------------------------
# Gráficos e salvamento
# -----------------------------------------------------------------------------

def save_bestfit_plot(model: CosmologyModel, data: HzData, theta_best: np.ndarray, outdir: Path) -> None:
    zgrid = np.linspace(np.min(data.z), np.max(data.z), 300)
    plt.figure(figsize=(7, 5))
    plt.errorbar(data.z, data.H, yerr=data.sigma, fmt="s", capsize=2, label="dados H(z)")
    plt.plot(zgrid, model.H_of_z(theta_best, zgrid), label="melhor ajuste")
    plt.xlabel(r"$z$")
    plt.ylabel(r"$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{model.name}_bestfit.png", dpi=180)
    plt.close()


def save_corner_plot(model: CosmologyModel, samples: np.ndarray, theta_best: np.ndarray, outdir: Path) -> None:
    fig = corner.corner(samples, labels=model.labels, truths=theta_best)
    fig.savefig(outdir / f"{model.name}_corner.png", dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Pipeline principal
# -----------------------------------------------------------------------------

def analyze_model(model: CosmologyModel, data: HzData, args: argparse.Namespace) -> dict[str, object]:
    print(f"\n=== Analisando {model.name} ===")
    outdir = Path(args.out)
    nwalkers = max(args.walkers_factor * model.ndim, 2 * model.ndim + 2)

    theta_map = find_map_with_de(model, data, seed=args.seed)
    print("Ponto inicial por DE:", theta_map)

    samples = run_mcmc(
        model=model,
        data=data,
        theta0=theta_map,
        nwalkers=nwalkers,
        nsteps=args.steps_post,
        burn_frac=args.burn_frac,
        thin=args.thin_post,
        seed=args.seed + 10,
        beta=1.0,
    )

    logL_values = np.array([log_likelihood(s, model, data) for s in samples])
    best_index = int(np.argmax(logL_values))
    theta_best = samples[best_index]
    logL_max = float(logL_values[best_index])
    q16, q50, q84 = np.percentile(samples, [16, 50, 84], axis=0)
    aic, bic = aic_bic(logL_max, k=model.ndim, n=len(data.z))

    # Grade mais densa perto de beta=0, como sugerido para integração termodinâmica.
    betas = np.unique(
        np.concatenate([np.linspace(0.0, 0.10, 5), np.linspace(0.15, 1.0, args.n_betas)])
    )
    logZ, ti_curve = thermodynamic_integration(
        model=model,
        data=data,
        theta0=theta_map,
        betas=betas,
        nwalkers=nwalkers,
        nsteps=args.steps_ti,
        burn_frac=args.burn_frac,
        thin=args.thin_ti,
        seed=args.seed + 100,
        n_prior_samples=args.prior_samples,
    )

    np.savetxt(outdir / f"{model.name}_posterior_samples.txt", samples, header=" ".join(model.labels))
    np.savetxt(outdir / f"{model.name}_ti_curve.txt", ti_curve, header="beta mean_logL_beta")
    save_bestfit_plot(model, data, theta_best, outdir)
    save_corner_plot(model, samples, theta_best, outdir)

    return {
        "model": model.name,
        "labels": model.labels,
        "theta_map": theta_map,
        "theta_best": theta_best,
        "q16": q16,
        "q50": q50,
        "q84": q84,
        "logL_max": logL_max,
        "AIC": float(aic),
        "BIC": float(bic),
        "logZ": logZ,
    }


def write_summary(results: list[dict[str, object]], outdir: Path) -> None:
    lines = ["# Resumo da análise", ""]
    for res in results:
        lines.append(f"## {res['model']}")
        lines.append(f"- logL_max: {res['logL_max']:.6f}")
        lines.append(f"- AIC: {res['AIC']:.6f}")
        lines.append(f"- BIC: {res['BIC']:.6f}")
        lines.append(f"- logZ por integração termodinâmica: {res['logZ']:.6f}")
        for label, med, lo, hi in zip(res["labels"], res["q50"], res["q16"], res["q84"]):
            lines.append(f"- {label}: {med:.6f} (+{hi-med:.6f}/-{med-lo:.6f})")
        lines.append("")

    if len(results) == 2:
        r1, r2 = results
        lines.append("## Comparação")
        lines.append(f"- Delta logZ = logZ({r1['model']}) - logZ({r2['model']}): {r1['logZ'] - r2['logZ']:.6f}")
        lines.append(f"- Delta AIC = AIC({r1['model']}) - AIC({r2['model']}): {r1['AIC'] - r2['AIC']:.6f}")
        lines.append(f"- Delta BIC = BIC({r1['model']}) - BIC({r2['model']}): {r1['BIC'] - r2['BIC']:.6f}")

    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ajuste bayesiano de LCDM e wCDM a dados H(z).")
    parser.add_argument("--data", default="data/Hz.csv", help="Caminho para o CSV com z,H(z),errH.")
    parser.add_argument("--out", default="results", help="Diretório de saída.")
    parser.add_argument("--seed", type=int, default=12345, help="Semente aleatória para reprodutibilidade.")
    parser.add_argument("--walkers-factor", type=int, default=12, help="nwalkers = fator * ndim.")
    parser.add_argument("--steps-post", type=int, default=20000, help="Passos MCMC para posterior final.")
    parser.add_argument("--steps-ti", type=int, default=2500, help="Passos MCMC por beta na TI.")
    parser.add_argument("--burn-frac", type=float, default=0.30, help="Fração inicial descartada como burn-in.")
    parser.add_argument("--thin-post", type=int, default=5, help="Thinning da posterior final.")
    parser.add_argument("--thin-ti", type=int, default=10, help="Thinning das cadeias temperadas.")
    parser.add_argument("--n-betas", type=int, default=10, help="Número de betas entre 0.15 e 1.0.")
    parser.add_argument("--prior-samples", type=int, default=30000, help="Amostras diretas do prior para beta=0.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_hz_data(args.data)
    results = [analyze_model(model, data, args) for model in default_models()]
    write_summary(results, outdir)

    print("\nArquivos salvos em:", outdir.resolve())
    print("Veja results/summary.md para o resumo numérico.")


if __name__ == "__main__":
    main()
