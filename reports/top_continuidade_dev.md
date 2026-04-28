# Top 20 municípios por continuidade política (dev)

> Revisão humana obrigatória. O briefing manda PARAR nesta etapa se
> os top 20 não fizerem sentido político — isso indica bug na lógica
> de continuidade.

Total de municípios cobertos: 100
Total de eleições municipais: 4

| # | UF | Município | Anos max mesmo partido | Anos max mesmo grupo | Eleições observadas | Última transição |
|---|----|-----------|------------------------|----------------------|---------------------|------------------|
| 1 | SP | Cajamar | 12 | 12 | 4 | total |
| 2 | SP | Cesário Lange | 12 | 12 | 4 | ruptura |
| 3 | SP | Florínia | 12 | 12 | 4 | ruptura |
| 4 | SP | Paulistânia | 12 | 12 | 4 | total |
| 5 | SP | Santana de Parnaíba | 12 | 12 | 4 | ruptura |
| 6 | SP | Taboão da Serra | 12 | 12 | 4 | ruptura |
| 7 | SP | Assis | 8 | 8 | 4 | ruptura |
| 8 | SP | Auriflama | 8 | 8 | 4 | total |
| 9 | SP | Balbinos | 8 | 8 | 4 | ruptura |
| 10 | SP | Barra do Turvo | 8 | 8 | 4 | ruptura |
| 11 | SP | Bilac | 8 | 8 | 4 | ruptura |
| 12 | SP | Bom Sucesso de Itararé | 8 | 8 | 4 | ruptura |
| 13 | SP | Caiabu | 8 | 8 | 4 | total |
| 14 | SP | Campos Novos Paulista | 8 | 8 | 4 | total |
| 15 | SP | Canitar | 8 | 8 | 4 | ruptura |
| 16 | SP | Corumbataí | 8 | 8 | 4 | ruptura |
| 17 | SP | Ibirá | 8 | 8 | 4 | ruptura |
| 18 | SP | Inúbia Paulista | 8 | 8 | 4 | ruptura |
| 19 | SP | Irapuru | 8 | 8 | 4 | ruptura |
| 20 | SP | Itapirapuã Paulista | 8 | 8 | 4 | ruptura |

**Interpretação:** `anos_max_mesmo_partido` = maior sequência de eleições em que o MESMO partido venceu no município × 4 anos. `anos_max_mesmo_grupo` estende considerando coligações sobrepostas (parcial conta como 2 anos).

**Ressalva:** em modo dev temos apenas eleições municipais cobertas pelo mapping presidencial→municipal ([2012, 2016, 2020, 2024]). Com histórico curto, municípios de 3ª/4ª coloção na dominância podem não aparecer.