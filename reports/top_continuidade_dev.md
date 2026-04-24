# Top 20 municípios por continuidade política (dev)

> Revisão humana obrigatória. O briefing manda PARAR nesta etapa se
> os top 20 não fizerem sentido político — isso indica bug na lógica
> de continuidade.

Total de municípios cobertos: 100
Total de eleições municipais: 3

| # | UF | Município | Anos max mesmo partido | Anos max mesmo grupo | Eleições observadas | Última transição |
|---|----|-----------|------------------------|----------------------|---------------------|------------------|
| 1 | SP | Cesário Lange | 12 | 12 | 3 | total |
| 2 | SP | Florínia | 12 | 12 | 3 | total |
| 3 | SP | Santana de Parnaíba | 12 | 12 | 3 | total |
| 4 | SP | Taboão da Serra | 12 | 12 | 3 | total |
| 5 | SP | Assis | 8 | 8 | 3 | total |
| 6 | SP | Balbinos | 8 | 8 | 3 | total |
| 7 | SP | Barra do Turvo | 8 | 8 | 3 | total |
| 8 | SP | Bilac | 8 | 8 | 3 | total |
| 9 | SP | Bom Sucesso de Itararé | 8 | 8 | 3 | ruptura |
| 10 | SP | Cajamar | 8 | 8 | 3 | total |
| 11 | SP | Canitar | 8 | 8 | 3 | ruptura |
| 12 | SP | Corumbataí | 8 | 8 | 3 | total |
| 13 | SP | Ibirá | 8 | 8 | 3 | ruptura |
| 14 | SP | Inúbia Paulista | 8 | 8 | 3 | total |
| 15 | SP | Irapuru | 8 | 8 | 3 | ruptura |
| 16 | SP | Itapirapuã Paulista | 8 | 8 | 3 | ruptura |
| 17 | SP | Itapura | 8 | 8 | 3 | total |
| 18 | SP | Paranapanema | 8 | 8 | 3 | ruptura |
| 19 | SP | Paulistânia | 8 | 8 | 3 | total |
| 20 | SP | Peruíbe | 8 | 8 | 3 | total |

**Interpretação:** `anos_max_mesmo_partido` = maior sequência de eleições em que o MESMO partido venceu no município × 4 anos. `anos_max_mesmo_grupo` estende considerando coligações sobrepostas (parcial conta como 2 anos).

**Ressalva:** em modo dev temos apenas eleições municipais cobertas pelo mapping presidencial→municipal ([2012, 2016, 2020]). Com histórico curto, municípios de 3ª/4ª coloção na dominância podem não aparecer.