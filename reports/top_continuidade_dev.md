# Top 20 municípios por continuidade política (dev)

> Revisão humana obrigatória. O briefing manda PARAR nesta etapa se
> os top 20 não fizerem sentido político — isso indica bug na lógica
> de continuidade.

Total de municípios cobertos: 5568
Total de eleições municipais: 8

| # | UF | Município | Anos max mesmo partido | Anos max mesmo grupo | Eleições observadas | Última transição |
|---|----|-----------|------------------------|----------------------|---------------------|------------------|
| 1 | PE | Itapetim | 28 | 28 | 7 | total |
| 2 | PE | Vertentes | 28 | 28 | 7 | total |
| 3 | RS | Itatiba do Sul | 28 | 28 | 7 | total |
| 4 | RS | Tio Hugo | 28 | 28 | 7 | total |
| 5 | PI | Joaquim Pires | 24 | 24 | 8 | total |
| 6 | PI | São João do Arraial | 24 | 24 | 8 | total |
| 7 | PI | Teresina | 24 | 24 | 8 | ruptura |
| 8 | PE | Afogados da Ingazeira | 24 | 24 | 7 | total |
| 9 | PE | Carnaíba | 24 | 24 | 7 | total |
| 10 | AL | Flexeiras | 24 | 24 | 7 | total |
| 11 | AL | São Sebastião | 24 | 24 | 7 | total |
| 12 | MG | São Gonçalo do Rio Abaixo | 24 | 24 | 7 | total |
| 13 | SP | Cravinhos | 24 | 24 | 7 | ruptura |
| 14 | SP | Ilha Comprida | 24 | 24 | 7 | ruptura |
| 15 | SP | Junqueirópolis | 24 | 24 | 7 | ruptura |
| 16 | SC | Sangão | 24 | 24 | 7 | total |
| 17 | RS | Catuípe | 24 | 24 | 7 | ruptura |
| 18 | RS | Chiapetta | 24 | 24 | 7 | total |
| 19 | RS | Doutor Maurício Cardoso | 24 | 24 | 7 | total |
| 20 | RS | Minas do Leão | 24 | 24 | 7 | total |

**Interpretação:** `anos_max_mesmo_partido` = maior sequência de eleições em que o MESMO partido venceu no município × 4 anos. `anos_max_mesmo_grupo` estende considerando coligações sobrepostas (parcial conta como 2 anos).

**Ressalva:** em modo dev temos apenas eleições municipais cobertas pelo mapping presidencial→municipal ([1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024]). Com histórico curto, municípios de 3ª/4ª coloção na dominância podem não aparecer.