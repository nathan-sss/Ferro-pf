# 参数物理手册 — 引擎中每一个 script 可见参数的物理原理

## 怎么读这本手册

一个参数一条词条,回答同样八个问题(SPEC §42 固定模板):**名称(键/参数名)| 符号 | 物理含义 |
进入的公式(SPEC § + `file:line`)| 单位(deck 与 SI,`units.py` 换算因子)| 典型范围与文献值 |
怎么选(仓库法则)| 陷阱**。§A–§F 以小标题起一条词条,§G–§J 以加粗行起,内容同一套模板。

**证据法(SPEC §42 EVIDENCE LAW,逐值标注)。没有来源的量一律写"仓库内无来源",绝不从记忆补值**,
并在文末附录里逐条列出(普查 T6 全表,24 条)。

| 标记 | 含义 |
|---|---|
| `[toml-meta]` | `params/*.toml` 的 `[meta]` 里有逐页核实的出处 |
| `[LIT]` | `reports/LIT_*.md` 的提取记录 |
| `[pdf]` | `literature/` 下有 PDF |
| `[std-ref 未复核]` | 有完整文献著录,但本仓库未再核对原件 |
| `[DIAL §x]` | 仓库自定的旋钮,无外部来源,注明声明它的 SPEC § |
| `[DIAL V2-02.1]` | 由 SPEC V2-02.1 裁定的单位映射(κ_deck = 0.0088541878128·ε_b、E_deck = E_SI×1e-9)**算出来**的数 —— 不是外部来源,也不是旋钮,推导在 `reports/REVIEW_v2_02_units.md` |

**§K 的材料值表是生成的**:每一格由 `tools/param_tables.py` 从 `params/*.toml` 与 `params/archive/*.toml` 读出
(重生成 `python3 tools/param_tables.py`,查过期 `--check`),所以不会与甲板脱节;物理解释是手写的。
**改数改甲板,不要改 §K。**

**甲板简称**(与 `reports/archive/v1/REVIEW_wp30_param_census.md` 一致):`bfoP`=bfo_pan2019 · `bto118`=bto_118K · `bto298`=bto_298K · `btoBC`=bto_BC1984_6th · `btoL8`=bto_LCC2005_8th · `btoLp`=bto_LCC2005_paper · `btoP`=bto_pan2019 · `nnbm`=nnbm_wang2026 · `pbzoK`=pbzo_kittel_lum2022 · `pbzoT`=pbzo_triplewell_yang2025 · `pmn0`=pmn_clean_x0 · `pmnD`=pmn_disorder_R3 · `pmnPT`=pmn_pt_landau_x0.30 · `pto`=pto_Haun1987 · `pzt06`=pzt_Haun_x0.6 · `pztG`=pzt_r3c_g22e_model · `pztH`=pzt_r3c_haun_x02 · `stoA`=sto_afd_rabeA · `stoP`=sto_pan2019。

**已存档的甲板**(2026-11,用户裁定:没有任何计算读取它们,`git mv` 到 `params/archive/`,文件名不变、数值未改;
逐套原因与恢复方法见 `params/archive/README.md`):`btoBC` · `btoL8` · `pmn0` · `pmnD` · `pmnPT` · `pto` · `pzt06` ·
`pztG` · `pztH`。本书照常保留它们的词条与 §K 数值,§K 各表中标「(存档)」。其中 `btoL8`/`pto`/`pzt06` 的
**生成函数仍在用**(`materials.get("bto_lcc2005_8th"/"pto_haun1987"/"pzt_haun_x")`),存档的只是 TOML 快照;
`btoBC`/`pmn0`/`pmnPT` 的生成函数随之移到 `solvers/materials_archive.py`(`materials_archive.get(name)`)。

## 目录

- **§A 单位制与换算** — A.0 什么叫 deck 单位 · A.1 基本单位 · A.2 `units.py` 的每一个换算函数 ·
  A.3 无需换算的量 · A.4 三个必须记住的坑 · A.5 关于"典型范围"的一句总纲
- **§B Landau 系数** — B.0 物理 · B.1 逐项词条(`alpha1`…`alpha1123`)· B.2 材料表 → §K ·
  B.3 在案的冲突与缺口
- **§C 梯度能** — C.0 物理 · C.1 词条(`G110`、四个比值、`H11`)· C.2 材料表 → §K ·
  C.3 缺口:梯度扇区没有任何文献范围
- **§D 静电** — D.0 物理 · D.1 词条(`permittivity`/`kappa_b`、短路约定、`E_ext`、κ(x) 死层 SPM)
- **§E 弹性与电致伸缩** — E.0 物理 · E.1 词条(`C11/C12/C44`、`Q11/Q12/Q44`、`k0`、`eps_m`、
  `e0`、`hf`/`hs_dx`)· E.2 什么时候开弹性 · E.3 材料表 → §K
- **§F 动力学与数值参数** — F.0 `L` 与总纲 · F.1 时间步 `dt`(一个探测器 + 三条法则)·
  F.2 有噪声时:`kT`、`noise_cutoff_nm`、判停、种子 · F.3 网格
- **§G 无序与淬火场** — G.1 取代型点电荷 · G.2 带电纳米区(默认关) · G.3 键序/COR ·
  G.4 声明式旋钮生成器 · G.5 成分场与 Vegard · G.6 复合相标签场 · G.7 PZT 系数场与缺陷场
- **§H AFE `q` 扇区** — `aq*`、`t1/t2`、`u1/u2`、`Gq`、`Lam*`、`kappa_b`、三阱闭式阈值、`loopfit` 靶
- **§I 八面体倾转 `θ` 扇区** — θ 的两套单位约定、`b*`、`t11/t12/t44`、`Gtheta`、`Lam*_theta`
- **§J 协议参数** — `build_case`、两条初态协议、`n1`/`e_max`/`n_dwell`、`energy_integrals`、
  `verified_dt`、J.1 薄膜通道常量
- **§K 材料值总表(生成)** — K.1 出处与文献支撑 · K.2/K.3 Landau · K.4 梯度 · K.5 介电 ·
  K.6 弹性 · K.7 电致伸缩 · K.8 `q` 扇区 · K.9 `θ` 扇区 · K.10 `H11`
- **附录:仓库内无来源的参数(普查 T6 全表,24 条)**

## 交叉索引:键 / 参数名 → 词条所在

按字母序,由脚本扫描全书的词条标题与小节标题生成(共 165 个名字)。表格里出现、但没有自己词条的数值旋钮,请查所在小节的参数表。

| 键 / 参数名 | 词条所在 |
|---|---|
| `a1` | §B B.1 |
| `a11` | §B B.1 |
| `a111` | §B B.1 |
| `a112` | §B B.1 |
| `a12` | §B B.1 |
| `a123` | §B B.1 |
| `a_lattice` | §G G.5 |
| `a_lattice_nm` | §G G.1 |
| `alpha1` | §B B.1 |
| `alpha11` | §B B.1 |
| `alpha111` | §B B.1 |
| `alpha1111` | §B B.1 |
| `alpha1112` | §B B.1 |
| `alpha112` | §B B.1 |
| `alpha1122` | §B B.1 |
| `alpha1123` | §B B.1 |
| `alpha12` | §B B.1 |
| `alpha123` | §B B.1 |
| `alpha_field` | §G G.6 |
| `alpha_field_multi` | §G G.6 |
| `amplitude` | §F F.3 / §J |
| `aq1` | §H |
| `aq11` | §H |
| `aq111` | §H |
| `aq112` | §H |
| `aq12` | §H |
| `aq123` | §H |
| `assemble_eps_star_static` | §G G.4 |
| `b1` | §I |
| `b11` | §I |
| `b111` | §I |
| `b112` | §I |
| `b12` | §I |
| `b123` | §I |
| `bond_order_alpha1_field` | §G G.3 |
| `build_case` | §J |
| `C11` | §E E.1 |
| `C12` | §E E.1 |
| `C44` | §E E.1 |
| `C_cw_K` | §G G.3 |
| `charge_disorder_field` | §G G.1 |
| `charge_disorder_stats` | §G G.1 |
| `chi_AFE` | §H |
| `composition_std` | §G G.5 |
| `composition_std_correlated` | §G G.5 |
| `conc` | §G G.1 |
| `defect_field` | §G G.7 |
| `delta_nm` | §J |
| `delta_q` | §G G.1 |
| `delta_q_region` | §G G.2 |
| `disorder.random_field_gaussian` | §G G.4 |
| `disorder_sigma_max_deck` | §G G.4 |
| `disorder_xi_erand_nm` | §G G.4 |
| `dlna_dx` | §G G.5 |
| `dt` | §F F.1 |
| `dt0` | §J |
| `dT0_region_K` | §G G.3 |
| `e0` | §E E.1 / §J J.1 |
| `E_A` | §H |
| `E_ext` | §D D.1 |
| `E_F` | §H |
| `E_max` | §H / §J |
| `e_max` | §J |
| `e_path_unipolar` | §J |
| `elastic_C_at` | §G G.7 |
| `elastic_C_field` | §G G.7 |
| `energy_integrals` | §J |
| `eps_m` | §E E.1 |
| `eps_r` | §G G.1 / §G G.6 |
| `f_region` | §G G.2 |
| `filler_alpha_deck` | §G G.6 |
| `formula_units_per_cell` | §G G.5 |
| `fractions` | §G G.6 |
| `G11` | §C C.1 |
| `G110` | §C C.1 |
| `G11_G110` | §C C.1 |
| `G12` | §C C.1 |
| `G12_G110` | §C C.1 |
| `G44` | §C C.1 |
| `G44_G110` | §C C.1 |
| `G44p` | §C C.1 |
| `G44P_G110` | §C C.1 |
| `gaussian_random_field` | §G G.4 |
| `Gq` | §H |
| `Gtheta` | §I |
| `h` | §F F.3 |
| `H11` | §C C.1 |
| `hf` | §E E.1 |
| `host_alpha_deck` | §G G.6 |
| `hs_dx` | §E E.1 / §J J.1 |
| `ic_clean` | §J |
| `ic_noise_yang` | §J |
| `k0` | §E E.1 |
| `kappa_b` | §D D.1 / §H / §J |
| `KAPPA_B_DECK` | §J J.1 |
| `KAPPA_B_PAPER_LINEAGE` | §J J.1 |
| `kappa_for` | §J |
| `kmax_frac` | §F F.3 |
| `kT` | §F F.2 |
| `L` | §F F.0 |
| `Lam11` | §H |
| `Lam11_theta` | §I |
| `Lam12` | §H |
| `Lam12_theta` | §I |
| `Lam44` | §H |
| `Lam44_theta` | §I |
| `lambda_s_nm` | §G G.1 |
| `landau_field` | §G G.7 |
| `loopfit.fit` | §H / §J |
| `make_label_field` | §G G.6 |
| `make_label_field_multi` | §G G.6 |
| `make_random_strain` | §G G.4 |
| `material_alpha_decks` | §G G.6 |
| `max_halvings` | §J |
| `mix_decks` | §G G.5 |
| `N` | §J |
| `n1` | §J |
| `n1_probe` | §J |
| `n_dwell` | §J |
| `n_dwell_probe` | §J |
| `n_equil` | §J |
| `P_max` | §H |
| `paper_alpha` | §J J.1 |
| `PAPER_DX_NM` | §J J.1 |
| `paper_grid` | §J J.1 |
| `PAPER_HF_DX` | §J J.1 |
| `PAPER_NZ` | §J J.1 |
| `pe_filler_landau_si` | §G G.6 |
| `permittivity` | §D D.1 |
| `probe_stable_dt` | §F F.1 |
| `Q11_phys` | §E E.1 |
| `Q12_phys` | §E E.1 |
| `Q44_phys` | §E E.1 |
| `Q_field` | §G G.7 |
| `rms` | §G G.7 |
| `run_unipolar` | §J |
| `scheme` | §F F.1 |
| `seed` | §F F.2 / §G G.1 / §G G.6 / §G G.7 / §J |
| `seeds` | §H / §J |
| `si_to_deck_gradient` | §H / §I |
| `si_to_deck_landau` | §G G.3 / §H |
| `si_to_deck_theta` | §I |
| `si_to_deck_theta_angle_deg` | §I |
| `std` | §G G.5 |
| `T` | §F F.2 / §G G.3 / §G G.7 |
| `T0_K` | §G G.3 |
| `t1` | §H |
| `t11` | §I |
| `t12` | §I |
| `t2` | §H |
| `t44` | §I |
| `T_total` | §J |
| `theta` | §I |
| `tol` | §H / §J |
| `u1` | §H |
| `u2` | §H |
| `verified_dt` | §J |
| `x` | §G G.7 |
| `x_AN` | §G G.6 |
| `x_bar` | §G G.5 |
| `xi` | §G G.6 / §G G.7 |
| `xi_AN` | §G G.6 |
| `xi_region_nm` | §G G.2 |
| `xi_x` | §G G.5 |
| `YANG_NOISE_AMPLITUDE_SI` | §J |

---

## §A 单位制与换算

### A.0 什么叫 "deck 单位"

`SPEC.md` §0 是权威原文:"**Units: adopt Ferret deck units verbatim. Never convert.** Lengths
nm, coefficients exactly as in the `.i` files. Mobility L=1"。这套单位不是本仓库发明的,是整体
继承 Ferret 算例(`.i` 甲板)的;`solvers/units.py:1-32` 的模块头把出处写死在
`FERRET_ROOT/runs/rhombo_walls/tools/gld1d.py` 自己的注释里——**能量密度 1e9 J/m³、长度 nm、
极化 C/m² 不缩放**。这三条一定,其余量纲全部由它们导出。

实际后果:甲板里的数**直接喂进引擎**,运行时不换算;换算只发生在"把文献的 SI 数写进
`params/*.toml`"这一步,由 `units.py` 的具名函数完成,核心门 item `K5.converter_roundtrip`
(吸收原 `gates/archive/g6a_converter_roundtrip.py`)是唯一权威测试(Landau 往返 ≤1e-8 相对)。

### A.1 基本单位

| 量 | deck 单位 | SI | 说明 |
|---|---|---|---|
| 长度 | 1 nm | 1e-9 m | `Lx/Ly/Lz`、`hx`、ξ、λ_s 一律 nm |
| 能量密度 | 1e9 J/m³ | J/m³ | **同时恰好等于 1 GPa**——所以 deck 的 `C_ij` 数值就是"GPa 数" |
| 极化 P | 不缩放 | C/m² | SI 与 deck 同一个数(`units.py:5-6`) |
| 电场 E | deck | V/m | `E_deck = E_SI × 1e-9`,由能量密度**导出**(不是自由约定);函数 `units.si_to_deck_field`(见 A.2/A.4) |
| 时间 | deck | — | 由 L=1 定死(SPEC §0);`dt` 无独立 SI 锚,只能数值验证(见 §F) |
| 2D 总能 F | [能量密度]×[长度]² = 1e-9 J/m | J/m | `grid['Volume']=Lx*Ly` 是**面积**,2D 的"能量"实为**单位 z 长度上的能量** |
| 畴壁能 σ_w | 能量密度×长度 = J/m² | J/m² | 1e9×1e-9=1,`wall_metrics` 乘 1000 纯粹是改名成 mJ/m² |
| θ(倾转)模式 A | nm-数 | m | `THETA_LEN_SCALE=1e-9`(`units.py:150`)——θ 是**线性氧位移** |
| θ(倾转)模式 B | 不缩放 | 度 | 逐甲板约定(`units.py:172-185`),**不是**全局约定 |

### A.2 `units.py` 的每一个换算函数

| 函数 | 行 | 因子 | SI 源单位 | 用在 |
|---|---|---|---|---|
| `SI_TO_DECK_ENERGY`(常量) | :33 | 1e-9 | J/m³ | 能量密度基准因子 |
| `si_to_deck_landau(v)` | :36 | ×1e-9 | J·m/C²(a₁) | **任意阶** Landau a_n,与 n 无关 |
| `si_to_deck_elastic(v)` | :42 | ×1e-9 | Pa | C₁₁/C₁₂/C₄₄;与 landau 同因子,分开命名只为调用点清晰(`:45-47` 原话) |
| `si_to_deck_gradient(v)` | :51 | **×1e9** | J·m³/C² | G₁₁₀/G₁₁/G₁₂/G₄₄/G₄₄′——**反方向**,推导见 `:60-81` |
| `kT_langevin(T,hx,hy,hz)` | :105 | k_B·T·1e18/(hx·hy·hz) | J | **朗之万 FDT 系数,三个引擎唯一的在案约定**(SPEC V2-04.0 / 裁决 D16);2D 的 hz 是涨落层厚 Dz(惯例 hx) |
| `si_to_deck_kT(T,hx,k_B)` | :162 | k_B·T/(hx·1e-9)×1e9 | J | k_BT → F_total 单位的换算,**不是** FDT 系数;V2-04.0 起已退出全部朗之万调用点 |
| `K_BOLTZMANN_SI` | :102 | 1.380649e-23 J/K | — | 2019 SI 重定义,精确值 `[std-ref 未复核]` |
| `si_to_deck_theta(v_m)` | :155 | ÷1e-9 | m | 模式 A 的 θ |
| `si_to_deck_theta_angle_deg(v)` | :172 | ×1(恒等) | 度 | 模式 B 的 θ,只为 API 对称与命名 |
| `si_to_deck_theta_landau(v,m)` | :196 | ×(1e-9)^m×1e-9 | 能量密度 | **仅模式 A**;m = 该项里 θ 的总幂次 |
| `cubic_compliance_to_stiffness(s11,s12,s44)` | :211 | s[m²/N]→c[Pa] | m²/N | `pto`、`pzt06`(源给的是柔度) |
| `cubic_stiffness_to_compliance(c11,c12,c44)` | :85 | c→s | Pa | 立方自逆式(Nye);WP10/§16 弹性 SPM 用 |
| `si_to_deck_field(v)` | :250 | ×1e-9 | V/m | `E_ext`、淬火 E 场;因子由"−P·E 必须是能量密度"**导出**(A.4 坑 3);反向 `deck_to_si_field`(`:261`)×1e9;常量 `EPSILON_0_SI`(`:247`) |
| `si_to_deck_permittivity(v)` | :266 | **×1e9** | F/m(绝对 κ) | κ_SI = ε₀ε_b → deck κ |
| `relative_to_deck_permittivity(eps_b)` | :275 | ×0.0088541878128 | 无量纲 ε_b | **写 `[electrostatics].permittivity` 的唯一正确入口**;0.0088541878128 = ε₀×1e9(见 A.3、§D、§K.5) |

### A.3 无需换算的量(SI == deck,SPEC §9.1(a))

Q₁₁/Q₁₂/Q₄₄ 与 Λ₁₁/Λ₁₂/Λ₄₄ `[m⁴/C²]`:应变无量纲、P 不缩放,SI 与 deck 之间没有任何长度或
能量密度幂次改变。梯度**比值** `G11_G110` 等本身无量纲。κ(`permittivity`)是 deck 原生量
(**deck ε₀ = 0.0088541878128** = ε₀_SI×1e9),真正的物理内容是
**乘在 ε₀ 上的倍数 ε_b = κ/0.0088541878128**(见 §D、§K.5)。**SPEC §1.3 举的 0.08854187 不是 deck ε₀,是 10ε₀**
——即 ε_b = 10,Ferret 自己那句 "e_b = 10" 的值;勘误的在案版本是 SPEC V2-02.1 `[DIAL V2-02.1]`。

### A.4 三个必须记住的坑

1. **梯度 ×1e9 vs Landau ×1e-9,方向相反。** 梯度系数分母多带一个 length²,SI→deck 净因子是
   1e18×1e-9 = 1e9(`units.py:60-75` 逐因子推导);把 landau 因子套到梯度系数上**差 1e18**。
   在案检查:Pan 2019 的 G₁₁₀ = 7.04e-11 SI → 0.0704 deck(`units.py:75-81`)。
2. **`si_to_deck_kT` 和 `bulk3d.kT_deck_3d` 都不是朗之万 kT**(SPEC V2-04.0 / 裁决 D16)。
   两者算术都对(k_B·T → 各引擎 `F_total` 的单位),但 FDT 系数必须是**能量密度**,而它们一个返回
   "每单位 z 长度的能量"、一个返回**能量**。三个引擎统一走 `units.kT_langevin`——完整词条见 §F.2
   的 `kT`,约定与勘误的在案表述见 `docs/CAPABILITIES.md` §4/§6.6。
3. **电场的 ×1e-9 是导出量,不是"纯约定";κ 的基数是 0.0088541878128,不是 0.08854187。**
   能量密度 1e9 J/m³ ÷ P 的 1 C/m² = 1e9 V/m = 1 V/nm ⇒ `E_deck = E_SI×1e-9`;`si_to_deck_landau`
   (a₁_deck = a₁_SI×1e-9)只有在这个因子下才自洽(P = E/(2a₁) 单位不变),所以它是**被逼出来的
   唯一解**。同一套推导给出 κ_deck = κ_SI×1e9 ⇒ **κ = 0.0088541878128·ε_b**。函数是
   `units.si_to_deck_field` / `relative_to_deck_permittivity`(V2-02.1 落地,见 A.2),门是 K5 的
   `permittivity_convention`(四条腿都用 SI 闭式,10× 错会当场爆)。nnbm 的
   `[meta].disorder_sigma_source`:0.10 deck ↔ 1e8 V/m(100 MV/m)`[LIT]` 仍是在案交叉验证。
   **普查 T6 #12 由 V2-02 关闭。**

### A.5 关于"典型范围"的一句总纲
**甲板里的数是有出处的数,不是"典型值"。** 凡普查 T6 列为缺口的量,本册只写"仓库内无来源",
并说明仓库自己怎么处理(借用/旋钮/占位)。想要"文献常见区间"而本册没写的,**就是真的没有**。

---

## §B Landau 系数 `a1..a123`、`a1111..a1123`

### B.0 物理

体自由能密度(SPEC §1.1,`solvers/solver.py:110` `bulk_energy_density`):

    f_bulk = α₁(P₁²+P₂²+P₃²) + α₁₁(P₁⁴+P₂⁴+P₃⁴) + α₁₂(P₁²P₂²+P₂²P₃²+P₁²P₃²)
           + α₁₁₁(P₁⁶+P₂⁶+P₃⁶) + α₁₁₂[P₁⁴(P₂²+P₃²)+…] + α₁₂₃P₁²P₂²P₃²

每组系数对应一族**立方不变量**:**单下标重复**的(α₁₁、α₁₁₁)只看单分量幂次,决定**沿一个轴**
的势阱形状;**混合下标**的(α₁₂、α₁₁₂、α₁₂₃)只在多分量同时非零时起作用,是**相选择项**——决定
基态是四方(T:⟨100⟩)、正交(O:⟨110⟩)还是菱方(R:⟨111⟩)。换一套 α₁₁₂/α₁₂₃ 会换相,
而 α₁ 只管"有没有极化"。

数值实现上有一处必须知道:力 `bulk_force`(`solver.py:117`)**不是** sympy 逐项求导的直译,而是
pow-free 的 Horner 型改写(所有幂次由 q_i=P_i² 经乘法达到,三分量共享的六个立方对称组只算一次;
WP28 T1/SPEC §40.3),与 sympy 版逐点等价(`solver.py:194` `_verify_bulk_force_horner` 在
import 时核对)。**改 Landau 项要改的是那个 Horner 式。**

### B.1 逐项词条
#### `alpha1` → 引擎键 `a1` · α₁ — 二阶项,Curie–Weiss 系数
- **物理含义** 顺电↔铁电的开关:α₁>0 时 P=0 是唯一极小(顺电),α₁<0 时 P=0 失稳、出现自发
  极化。温度依赖是 Curie–Weiss 式 **α₁(T) = (T−T₀)/(2ε₀C)**(T₀ = Curie–Weiss 温度,C = Curie
  常数),这也是仓库唯一显式实现的 α₁(T) 形式(`quenched_fields.py:359` `bond_order_alpha1_field`,
  把 T₀ 做成空间场即得无序;SPEC §9.1(d) 的 PZT 生成式同式)。|α₁| 同时定标 ξ ∝ √(G/|α₁|)。
- **进入的公式** SPEC §1.1;`solvers/solver.py:110`、`:117`。
- **单位** deck `1e9 J/m³/(C/m²)²` · SI `J·m/C²` · `units.si_to_deck_landau` ×1e-9。
- **范围/文献值** 见 B.2 材料表。全部 17 套有 Landau 扇区的甲板都有它;`pmn0` **只有**它。
- **怎么选** 永远从甲板取、不跨算例复制(SPEC §0);换温度用 `materials.get(name, T=…)` 让生成
  式重算,不要手改 α₁。
- **陷阱** α₁ 可以是**空间数组**(SPEC §1.1 原文 "α1 may be a spatial array"),无序/复合算例正是
  这么用的;凡是假设 α₁ 是标量的下游代码都会在这里坏掉。

#### `alpha11` → `a11` · α₁₁ — 四阶单轴项,决定相变级数
- **物理含义** 单分量四次项。α₁₁ **>0**:势阱在 α₁ 过零处连续张开 → **二级(连续)相变**;
  α₁₁ **<0**:四次项与 α₁ 一起把 P=0 附近压塌,必须靠六阶 α₁₁₁ 兜住 → **一级相变**,P_s 跳变、
  伴随热滞与双稳。**看一眼 α₁₁ 的符号就知道这套系数描述哪类相变**——读甲板最快的判据
  (`bto298` −0.64755 → 一级,与 BaTiO₃ 事实相符;`stoA` +1.7 → 连续型)。
- **进入的公式** SPEC §1.1;`solver.py:110`、`:117`。
- **单位** 同 α₁(`si_to_deck_landau` ×1e-9)。
- **范围/文献值** 见 B.2。16 套甲板有(除 `pmn0`/`pmnD`/`pmnPT`)。
- **怎么选** 与 α₁₁₁ **成套**取,不可单换:α₁₁<0 而 α₁₁₁ 缺失或 ≤0 时自由能**无下界**,算例必发散。
- **陷阱** PMN 的 α₁₁ 是**故意缺失**的,不是 0(见 B.3 冲突条 #4)。

#### `alpha12` → `a12` · α₁₂ — 四阶交叉项,四阶相选择
- **物理含义** 只在两个分量同时非零时出现。α₁₂ 相对 2α₁₁ 的大小决定四阶层面偏好"单轴"还是
  "多轴":α₁₂ > 2α₁₁ 时把极化推向单轴⟨100⟩,反之偏好⟨110⟩/⟨111⟩。它是 T–O–R 竞争的第一层。
- **进入的公式 / 单位** 同 α₁₁。
- **范围/文献值** 见 B.2(16 套)。
- **怎么选 / 陷阱** 与 α₁₁ 同源同套。注意 `pbzoK` 与 `pbzoT` 的 α₁₂ 一个是 0、一个是 2.3145——
  这两套是**不同模型**(精确 Kittel 变换 vs 三阱工程甲板),不能互相借项。

#### `alpha111` → `a111` · α₁₁₁ — 六阶单轴项,稳定项
- **物理含义** 一级相变甲板的"兜底":α₁₁<0 时,只有 α₁₁₁>0 才能让 f_bulk 在大 P 处重新上翘,
  给出有限的 P_s 和有限的势垒。它同时决定一级相变的**势垒高度**,因而直接决定矫顽场量级。
- **进入的公式 / 单位** 同上。
- **范围/文献值** 见 B.2;`btoP` 的 α₁₁₁ 有一条**未裁决的冲突**,见 B.3 #1。
- **怎么选** 见"陷阱";永远与 α₁₁ 配套。
- **陷阱** 六阶系数是甲板间**最不一致**的一项(`bto298`/`btoBC` 8.004、`btoP` 5.52、
  `btoL8` 1.294,全是 BaTiO₃——不同作者用不同阶截断拟同一批数据)。`btoBC` 原为 −7.0656:
  Rabe App A 表中 α₁₁₁ 的参考温度 120 是摄氏度(393 K),按开尔文代入后六阶势无下界;已改为 T−393 K。**跨甲板借 α₁₁₁ 是最常见的
  物理错误。**

#### `alpha112`,`alpha123` → `a112`,`a123` · α₁₁₂, α₁₂₃ — 六阶混合项,低温相选择
- **物理含义** 六阶层面的相选择项。α₁₁₂ 惩罚(或奖励)"一个分量大、另两个小"的构型,α₁₂₃ 只在
  **三个分量同时非零**(即⟨111⟩菱方相)时才非零——它是唯一能单独区分 R 相的项。BaTiO₃ 的
  T→O→R 级联在这套模型里就靠 α₁₁₂/α₁₂₃ 的相对大小实现。
- **进入的公式 / 单位** 同上。
- **范围/文献值** 见 B.2(16 套)。
- **怎么选** 若算例目标是某个特定低温相(如 `pztG`/`pztH` 的 R3c),这两项**必须**来自同一套源,
  否则相图会错。
- **陷阱** `btoL8` 的 `[meta].or_transition_note` 直说 LCC2005 这套系数**不适合做 O–R 转变**——
  甲板给了数不等于这套数在你关心的相边界上可用。

#### `alpha1111`,`alpha1112`,`alpha1122`,`alpha1123` → `a1111..a1123` — 八阶项
- **物理含义** 八阶截断,只有 BaTiO₃ 的 LCC2005 系族用:让模型在**宽温区**(尤其接近一级相变、
  P 较大处)保持定量精度——六阶截断在那里有系统性偏差。
- **进入的公式** SPEC §1.1;`solver.py:117`,键名表 `solver.py:102` `ALPHA_KEYS_8TH`。
  **默认 0.0**,只有六阶甲板的旧算例不必提供它们。
- **单位** 同 α₁(与阶数无关,`units.py:14-18` 明确)。
- **范围/文献值** 12 套甲板带这四个键,**只有 `btoL8`/`btoLp` 非零**:38.63/25.29/16.37/13.67
  `[std-ref 未复核]`(Rabe/Ahn/Triscone 2007 App A(Chen)Table 1 "Eight-order",书 p.365)。
- **怎么选** 要么整套用八阶甲板,要么整套用六阶;**不要把八阶项加到别人的六阶拟合上**——那套
  六阶系数是在"八阶=0"的前提下拟出来的。
- **陷阱** SPEC §1.1 原文要求"Verify the 8th-order deck terms are zero for each case";引擎用
  `alpha.get(k, 0.0)` 静默补 0,所以**漏传八阶项不会报错,只会悄悄变成六阶模型**。

### B.2 材料表(Landau)→ §K

逐甲板的 `a1..a123` 与八阶四项,连同每一格的源级标注,由 `tools/param_tables.py`
从 `params/*.toml` 生成:**见 §K 表 K.2(6 阶)与表 K.3(8 阶,只列非零者)**。

### B.3 在案的冲突与缺口(按 SPEC §42:呈现,不裁决)

1. **`btoP` 的 α₁₁₁ = 5.52(T6 #18)。** 该甲板六项中五项与 Rabe App A "Sixth order" **逐位相同**,
   唯独 α₁₁₁ 与 Bell & Cross (1984) 世系分道扬镳。机理已由 `reports/LIT_pan2019_forensics.md`
   查清,**但取舍从未做出**。本手册按普查要求**并列呈现两侧,不替你选**;算例若对六阶敏感,
   必须自己声明用哪一支并说明理由。
2. **STO 的 α₁ 常数 35.5 K vs 36.5 K(T6 #19)。** Rabe App A 印 7.06(T−35.5),Pan 2019 印
   7.06e5(T−**36.5**),两侧均 300 dpi 确认——Pan 侧一处真实的转录差异。后果:`stoA` 与 `stoP`
   的 α₁ **按构造就不相等**(0.183166 vs 0.184619)。
3. **STO Table-3 逐行归属(T6 #20)。** Rabe App A 对整表统一引 refs [10]–[15],**任何单个 STO
   系数都无法归到具体某篇文章**。
4. **PMN 的 α₁₁ 及所有高阶(T6 #9)。** `pmn0` **故意不填**(不是填 0);仓库里唯一的数是
   `pmn_field.py:508` `ALPHA11_PMN_BRACKET = (−0.3, 0.3)`,其注释自陈 "order-of-magnitude ONLY …
   not PMN-derived"——**这不是 PMN 的文献范围**。
5. **PMN-PT 0.30 整套甲板(T6 #8):每个扇区都空。**
6. **`bto298`/`bto118` 的出处(T6 #21)。** 权威是 **Ferret 甲板文件**不是论文——仓库自己的基准
   甲板**没有文献范围**。

**范围:仓库内无来源** —— 上述 3–6,以及"某材料 Landau 系数的一般区间"这个问法本身:本仓库
从不记录区间,只记录**逐甲板的具体来源**。

---

## §C 梯度能 `G11/G12/G44/G44p`(与 `G110`、比值)+ 四阶梯度 `H11`

### C.0 物理

梯度项是**畴壁的全部代价来源**。SPEC §1.2:

    f_grad = ½G₁₁(P₁,₁²+P₂,₂²+P₃,₃²) + G₁₂(P₁,₁P₂,₂+…)
           + ½G₄₄[(P₁,₂+P₂,₁)²+…] + ½G₄₄′[(P₁,₂−P₂,₁)²+…]

k 空间算子(`solvers/solver.py:264` `build_operators`,与 SPEC §1.2 逐项对应):
A₁₁ = G₁₁k₁² + (G₄₄+G₄₄′)k₂²,A₂₂ = G₁₁k₂² + (G₄₄+G₄₄′)k₁²,A₁₂ = (G₁₂+G₄₄−G₄₄′)k₁k₂,
A₃₃ = (G₄₄+G₄₄′)(k₁²+k₂²),A₁₃=A₂₃=0。

**两条定标律**(定性判据,不是仓库的门):畴壁宽度 **ξ ∝ √(G/|α₁|)**、畴壁能
**σ_w ∝ √(G·|α₁|³)/|α₁₁|**。前者说明 G 与 α₁ 只以**比值**进入长度尺度——**这正是 `G110 = 1.0`
这种"重整化参考值"能用的原因**:把 G 定死为 1,长度单位改由 α₁ 定标,物理不变,只是不能再
声称 G 是测量值。后者说明 σ_w 对 G 只是平方根敏感、对 α₁ 是 1.5 次方敏感。网格必须解析 ξ,
仓库的做法见 SPEC §13.0 原文 "Δx = sqrt(G110/alpha0) as in WP6 Tier 2"——同一条律反用。

### C.1 词条
#### `G110` — 梯度参考标度
- **物理含义** 一个**标度因子**,不是独立物理量:甲板不直接给 G₁₁/G₁₂/G₄₄/G₄₄′,而给 `G110`
  加四个无量纲比值,读取时相乘得绝对值(`materials.py:117` `_flat_to_deck`,绝对值化在 `:140`;
  `G110` 与四个比值另留一份到 `meta` 作溯源)。
- **进入的公式** 只以乘积形式进入 `solver.py:264`(SPEC §1.2);3D 走 `bulk3d.build_operators_3d`
  (`bulk3d.py:165`),H11 车道走 `grad4.py:102`。
- **单位** deck `1e9 J/m³·nm²/(C/m²)²` · SI `J·m³/C²` · `units.si_to_deck_gradient` **×1e9**。
- **范围/文献值** 14 套甲板有(`pto`、`pzt06` **没有**,见 C.3)。唯一**有出处**的绝对值是
  Pan 2019 的 G₁₁₀ = 7.04e-11 SI → **0.0704 deck** `[pdf]`(`bfoP`/`btoP`/`stoP`)。
  `bto298`/`bto118` 及其派生的 **G₁₁₀ = 1.0 是重整化参考值,`units.py:54-58` 原文如此声明**
  `[DIAL §9.1(a)]`。`btoLp` 的 **0.00593856** 是由 dx=0.4 nm **反推**出来的
  `[toml-meta]`——其 `[meta]` 自称 "a documented free parameter, NOT tuned"。
  `nnbm` 的 **−0.664019**(**负值**)是与 H11 联合标定的旋钮 `[DIAL §32]`。
- **怎么选** 若甲板给了 `G110` 就用甲板的;若没有(`pto`/`pzt06`),算例**必须自己声明**一个值
  并标为占位假设(SPEC §9.1 结尾原文:"leave a REQUIRED field the case must set, with the
  BTO-normalized default clearly marked as a placeholder assumption")。
- **陷阱** 甲板读出的 `deck["gradient"]` **已乘过 G110**;`casekit.py:82` `build_2d` 的
  docstring 把"再手乘一次"点名为 FAQ #1 接线 bug。

#### `G11_G110`,`G12_G110`,`G44_G110`,`G44P_G110` → `G11`,`G12`,`G44`,`G44p`
- **物理含义** 梯度能的**各向异性指纹**。G₁₁ 罚"纵向"梯度(∂P₁/∂x₁ 这类),G₄₄ 罚剪切型梯度
  (∂P₁/∂x₂+∂P₂/∂x₁),G₄₄′ 罚其反对称组合(旋度型)。三者的相对大小决定 180° 壁与 90° 壁的
  能量比,进而决定畴图案的取向偏好。G₁₂ 是不同分量纵向梯度的交叉项。
- **进入的公式** SPEC §1.2;`solver.py:264`。
- **单位** 无量纲比值(乘 `G110` 后才有量纲);`units.py` 无对应函数。
- **范围/文献值** 16 套甲板有。两组在案:Pan 2019 的 **1.2 / 0 / 0.6 / 0.6** `[pdf]`
  (`bfoP`/`btoP`/`stoP`);BTO 的 **0.51 / −0.02 / 0.02 / 0**(源自 Ferret 甲板),被**借用**进
  `pto`、`pzt06`、`pbzoK`、`pbzoT`、`pztG`、`pztH`、`stoA` 七套非 BTO 甲板——每套自己的
  `[meta].note` 都写明"not X-validated"。
- **怎么选** 各向同性归约:`G12=0, G44=G44p=0.5*G11` ⇒ A₁₁=A₂₂=A₃₃=G₁₁₀k²、A₁₂=0
  (`btoLp` 的 1.0/0/0.5/0.5 与 `nnbm` 用的就是这一组;`grad4.py:102` docstring 逐字复用了
  `btoLp` 的这段说明)。薄膜车道(SPEC §13.0)按论文用各向同性 G。
- **陷阱** **借来的比值不是该材料的比值。** 见 C.3 缺口 #2。

#### `H11` — 四阶梯度系数
- **物理含义** 在梯度能之上再加一层 **∂²P** 的代价。当 **G₁₁ < 0** 时长波模式反而**增益**
  (体系想造周期结构),必须由 H₁₁ > 0 在大 k 处止住——两者共同挑出一个**有限的最优波数**,
  给出条纹/反铁电带。这就是 `grad4.py` 存在的理由。
- **进入的公式** SPEC §32;`solvers/grad4.py:102` `build_operators_hog`(在 A₁₁/A₂₂/**A₃₃**
  的对角乘子上加 `H11*(KX⁴+KY⁴)`),能量 `grad4.py:146` `hog_energy_density`
  (f_hog = ½H₁₁Σᵢ[(∂²Pᵢ/∂x²)²+(∂²Pᵢ/∂y²)²],SPEC §32.4 R1 **逐分量**形式,**不是**各向同性的
  (k²)²)。色散与选长:`grad4.py:226` `dispersion_rate`、**`grad4.py:232` `k_star`
  给出 k\* = √(−G₁₁/(2H₁₁))**(该函数在 G₁₁<0<H₁₁ 之外直接抛错),`grad4.py:241`
  `lambda_star` 给 λ\* = 2π/k\*。
- **单位** deck `1e9 J/m³·nm⁴/(C/m²)²`;`units.py` **无换算函数**,甲板里直接是 deck 数。
- **范围/文献值** **只有 `nnbm` 有:H11 = 22.046** `[DIAL §32]`。其 `[meta].gradient_dial_basis`
  写明:两篇源论文**都不发表任何梯度数值**("exhaustive grep, zero hits both"),G₁₁₀ 与 H₁₁ 是对
  λ\*≈51.2 nm 目标**联合标定**出来的。
- **怎么选** 有目标周期就用 k\* 反解;没有目标周期,**这个参数无从选起**。
- **陷阱** 逐分量的 kx⁴+ky⁴ 形式带**取向偏好**(等 |k| 圆上四次惩罚在 kx=ky 处最小 ⇒ 最快增长
  模沿 ±45°);SPEC §32.5 把"观测到的对角条纹"记为该形式正确的独立证据。换成各向同性形式
  会丢掉这个取向——**这不是等价改写。**

### C.2 材料表(梯度)→ §K

`G110`、四个比值,以及引擎实际用的绝对 `G11/G12/G44/G44p`(甲板里不存在,读取时由
`materials._flat_to_deck` 相乘得到)一并生成:**见 §K 表 K.4**。`H11` 见表 K.10。

### C.3 缺口:梯度扇区没有任何文献范围

- **#1(T6 #1)G₁₁/G₁₂/G₄₄/G₄₄′ 一般范围:仓库内无来源。** 唯一有出处的梯度数是 Pan 2019 的
  G₁₁₀=7.04e-11 SI;`G110=1.0` 是重整化参考值(`units.py:54-58` 自陈),**不是测量值**。
- **#2(T6 #2)BTO 比值组(0.51/−0.02/0.02/0)对 PTO/PZT/PbZrO₃/STO:仓库内无来源。** 七套甲板
  是借的,每套 `[meta].note` 自陈 "not X-validated"。
- **#3(T6 #3)H₁₁:仓库内无来源**(无任何已发表值或区间)。**#4(T6 #4)G_q 与 G_θ:仓库内
  无来源**(两者都等于 1.0,纯借用,SPEC §23.5(d)/§28.5;详见 §H `Gq` 与 §I `Gtheta`)。

---

## §D 静电:`kappa_b`、`E_ext`、κ(x) 与死层 SPM

### D.0 物理

SPEC §1.3:∇·(κ∇φ) = ∇·P,κ := 甲板的单个 `permittivity` 值(ε₀ε_b **合并后**的量)。k 空间
φ̂ = −ik_jP̂_j/(κk²),**Ê_d,i = −k_ik_jP̂_j/(κk²)**,退极化算子 D_ij = k_ik_j/k²;
f_elec = −½P·E_d − P·E_ext,力 = −E_d,i − E_ext,i(自场的 ½ 求导时消失,门 G0 查这一条)。

κ 的角色:**除显式建模的 P 之外**、由电子/离子快速响应贡献的**背景**屏蔽。κ 越大,退极化场
E_d = −D·P/κ 越弱。**κ 不是材料的介电常数**——总介电响应还含 P 自身的软模贡献,那部分归 Landau。

### D.1 词条
#### `permittivity` → 引擎键 `kappa_b` · κ
- **物理含义** 背景介电常数 ε_b 与真空 ε₀ 的乘积,退极化场的分母。
- **进入的公式** SPEC §1.3;`solver.py:264` `build_operators`(D(k)/κ)、`:425` `elec_Ed`、
  `:442`、`:469`;隐式矩阵也含它(SPEC §2:M=I+Δt·L·[A+D/κ]);薄膜 `film3d.py:338`。
- **单位** **deck 原生**:**deck ε₀ = 0.0088541878128**(= ε₀_SI×1e9),所以
  **κ = 0.0088541878128·ε_b**、**ε_b = κ/0.0088541878128** `[DIAL V2-02.1]`。
  **SPEC §1.3 举的 0.08854187 不是 deck ε₀,而是 10ε₀** —— 它是 Ferret 自己的 BTO 值,原注释
  写着 "e_b = 10, see PRB. 74, 104014, (2006)"
  (`$FERRET_ROOT/ferret/examples/domain_walls/BTO_wall_T298K.i:479-488`);WP1 导入时数字留下、
  那句注释丢了,Part I 的整条世系因此**低标 10 倍**。勘误的在案版本是 **SPEC V2-02.1**,推导与
  三个引擎的数值确认见 `reports/REVIEW_v2_02_units.md`。换算函数:
  `units.relative_to_deck_permittivity(ε_b)`(= 0.0088541878128·ε_b)、`si_to_deck_permittivity`
  (×1e9),见 §A.2。**甲板里的数一个没改(D7)——改的只是标签。**
- **范围/文献值 —— 旧标 1/10/50/66 的世系,真值是它的 10 倍**(每一档的源级不同,必须逐档看;
  逐甲板的 deck κ 与真 ε_b 见 §K 表 K.5)。**deck κ 列是仓库里真实跑过的数,一个没动**;
  "真 ε_b" 列 = κ/0.0088541878128 `[DIAL V2-02.1]`:
  | 旧标 ε_b | **真 ε_b** | deck κ | 用在 | 源级(**是这一列在说那个文献数字,不是甲板实跑值**) |
  |---|---|---|---|---|
  | 1 | **10** | 0.0885419 | `bto118/bto298/btoBC/btoL8/btoLp/pbzoK/pbzoT*/pztG/pztH/stoA` | **不是"HM2006 占位值"** —— 就是 Ferret BTO 甲板的 ε_b=10(`film3d.py:272` 旧注释 "HM2006-inherited placeholder" 说反了);`[DIAL §13]` |
  | 10 | **100** | 0.885419 | `bfoP/btoP/stoP/nnbm` | `[std-ref 未复核]` Wang, Ma, Li, Britson & Chen, Acta Mater **61** (2013) 7591, p.7597 **给的是 ε_b=10**,而甲板乘的是已经等于 10ε₀ 的基数;代码常量 `materials.py:445` `PAN2019_KAPPA_B_MULT=10.0` |
  | 50 | **500** | 4.42709 | `pbzoT`;`energy_protocol.py:43` `_KAPPA_B_DEFAULT=50.0` | **括号内的旋钮**——Yang 2025 正文只**符号性**地定义 K,从未给数(`[DIAL §27.3(b)/§28.7 R6]`);同一算例的 PE 填充相却按真 ε_r=50 走 SI 换算,**一个算例内差 10 倍**(U2,见 `reports/archive/v1/B1_REPORT.md` 的 V2-02.1 caveat) |
  | 66 | **660** | 5.8437… | 论文复现算例;`film3d.py:273` `KAPPA_B_PAPER_LINEAGE = 66.0*KAPPA_B_DECK` | `[std-ref 未复核]` Haun 1987 / 该论文 ref 25 **给的是 ε_b=66**;66× 乘在 10ε₀ 上 ⇒ 实跑 660,即退极化比论文世系弱 10 倍 |
  裁决在案(`btoLp` 的 `[meta].kappa_b_ruling`):**旧门 G0–G10 保留甲板原值(旧标 ε_b=1 = 真 10),
  论文复现算例用 66×(= 真 660)**;`film3d.py:264-271` 注释写明这是 SPEC §13.0 "if genuinely
  unstated … do NOT silently pick one" 的执行结果——甲板原值是默认,66× 是**并列报告的敏感性对照**。
  **用户裁决 D15:不重跑;在案结果继续以 660 / 500 的标签存在**(`docs/CAPABILITIES.md` §6.6)。
- **怎么选** 按算例目的选,并在算例里写明选了哪一档、为什么:复现论文 → 用该论文世系那一档;
  仓库内部对照(门/回归)→ 用甲板原值,否则打破历史基线。
- **陷阱** 两个入口语义不同:`materials`/`casekit` 路线的 `kappa_b` 是**已乘好的 deck κ**,而
  `energy_protocol.build_case(kappa_b=…)` 收的是**倍数**(内部 `energy_protocol.py:113`
  `kappa_for` 再乘基常数 0.08854187,而那个基常数本身 = 10ε₀)。**把 4.42709 传给 build_case
  得到的是 `kappa_for(4.4270935)` = 0.391983137** —— 即预期 κ 的 **0.08854187 倍**(小 11.3 倍),
  不是旧文写的 "50×50 倍"(SPEC §42.1 F3 的算术在 V2-02.1 更正)`[DIAL V2-02.1]`。
  V2-02 之后 `build_case` 另有两个不含歧义的入口:`eps_b=`(SI 相对介电常数)与
  `permittivity_deck=`(已换算好的 deck κ),见 §J。

#### 短路约定:Ê_d(k=0) = 0
- **物理含义** 均匀模的退极化场被强制为零 = **理想短路电极/完全补偿的表面电荷**:整体极化不
  产生宏观退极化场,只有极化的**空间不均匀部分**付静电代价。这也是周期算例能有单畴基态的原因。
- **进入的公式** SPEC §1.3 原文 "Ê_d(k=0)=0";`solver.py:285-287` 的
  `np.where(k2 == 0.0, 0.0, …)` 就是它。薄膜车道换成真边界条件(SPEC §13.0:φ=0 于膜上下表面)。
- **怎么选 / 陷阱** 这不是可调参数,是**约定**。要研究开路/部分屏蔽必须换车道(薄膜引擎),
  **不能靠调 κ 模拟开路**。

#### `E_ext` — 外加电场
- **物理含义** 施加场,f_elec 里的 −P·E_ext 项。实验上就是 **E = V/d**(极板电压除以样品厚度)。
  它是所有 P–E 回线、能量存储、开关协议的驱动量。
- **进入的公式** SPEC §1.3;`solver.py:442`/`:469`/`:550` `step`,3D `film3d.py:343`。
- **单位** deck E = **SI(V/m)× 1e-9**,这个因子是从 f_elec 必须是能量密度**导出**的,不是自由
  约定(A.4 坑 3)。在案交叉验证:`nnbm` 的 `[meta]` 记 0.10 deck ↔ 1e8 V/m = 100 MV/m `[LIT]`。
  **函数是 `units.si_to_deck_field`(反向 `deck_to_si_field`),V2-02.1 落地;门是 K5
  `permittivity_convention` 的 Leg C —— 普查 T6 #12 已关闭。**
- **范围/文献值** 无通用范围,量级由矫顽场决定。在案两组:`nnbm` 无序场 σ_max = **0.218 deck
  = 2.18 MV/cm** `[LIT]`;`pbzoT` `[meta].e_thresholds_closed_form` 的 E_F=6.687e−2、
  E_A=3.632e−2 deck `[LIT]`。
- **怎么选** 先用甲板自带的阈值(若有)定标扫场范围;**不要用"典型 kV/cm"猜**。
- **陷阱 1** 默认标量三元组 `(0.,0.,0.)`,但它**也可以是场**((3,Ny,Nx),`pzt_field.py:262`
  `defect_field` 就返回这种)——无序车道正是经 `E_ext` 通道注入随机场。
- **陷阱 2(SPEC §32.5,强制)** `probe_stable_dt` **必须带生产用的 E_ext 一起探**:在案事故里
  E_ext=0 的探测把稳定 dt 高估 **64×**,生产跑 NaN。

#### κ(x):非均匀介电与死层 SPM
- **物理含义** 让 κ 成为空间场,建模**死层/界面介电层**:电极附近介电常数低得多的一层承担了
  大部分电压降,在铁电层里留下反向退极化场——薄膜矫顽场增大与极化"钉扎"的主因之一。
- **进入的公式** SPEC §11.1(Wang 等 Acta Mater **61** (2013) 7591 §2 的固定点迭代)。
  `solver.py:264` 收到 (Ny,Nx) 数组时自动分裂 **κ₀=(κ_min+κ_max)/2** 与 dκ=κ(x)−κ₀,
  迭代在 `solver.py:370` `_spm_iterate`,判据 max|E^n−E^(n−1)|/max|E| ≤ tol。
- **数值参数** `spm_tol=1e-10`、`max_iter=200`、`warn_iter=80`(`solver.py:370`,SPEC §11.1
  原文 "cap 200 (warn >80)")`[DIAL §11.1]`;`solver.py:425/:442/:469/:480` 各带 `spm_tol=1e-10`。
- **在案的死层配置** SPEC §11.1 原文:κ_die = **10×** κ_FE,α₁_die = **+8.0**、高阶为 0,
  死层里 P 也有定义(线性介电层),对应 `runs/film32_khach/inputs/eps00_ctrl.i`
  `[DIAL §11.1]`——SPEC 声明的配置,不是文献测量。
- **怎么选 / 陷阱** 标量 κ 与数组 κ 走**不同代码路径**:标量时 `ops['dkappa']=None`,SPM 分支
  全关、与 WP5 前逐位相同、零开销;传数组即便内容均匀也会打开迭代。另:**E_ext 不参加 SPM
  迭代**(`solver.py:370` docstring)——死层不会"重新分配"外加场,这是刻意的约定。

---

## §E 弹性与电致伸缩:`C11/C12/C44`、`Q11/Q12/Q44`、`k0`、`eps_m`、`e0`、`hf/hs`

### E.0 物理

极化通过**电致伸缩**产生本征应变 ε\*(P)(SPEC §1.4):

    ε*₁₁ = Q₁₁P₁² + Q₁₂(P₂²+P₃²)   (循环)      ε*₁₂ = ½Q₄₄P₁P₂   (同理 ε*₁₃, ε*₂₃)

不同取向的畴有不同的 ε\*,晶格必须协调这些不匹配,代价就是弹性能——这是 90° 畴壁、畴图案取向
选择、薄膜"应变工程"的**全部机理**。求解走 Khachaturyan 零阶解(均匀 C 时**精确**):
τ̂ = C:ε̂\*,Ω⁻¹_ik = C_ijkl k_j k_l,û = −iΩ k τ̂。

### E.1 词条
#### `C11`,`C12`,`C44` — 立方弹性刚度
- **物理含义** 立方晶体的三个独立刚度。C₁₁ 纵向、C₁₂ 横向耦合、C₄₄ 剪切。它们决定晶格"抵抗"
  本征应变的硬度,因而决定弹性能相对 Landau 能的权重。
- **进入的公式** SPEC §1.4(σ₁₁=C₁₁ε₁₁+C₁₂(ε₂₂+ε₃₃);σ₁₂=2C₄₄ε₁₂);`elastic.py:75`
  `build_C_tensor`、`:181` `setup_elastic`;3D `bulk3d.setup_elastic_3d`;薄膜 `film_elastic.py:301`。
- **单位** deck 数**就是 GPa 数**(1 deck 能量密度单位 = 1e9 J/m³ = 1 GPa);SI Pa;
  `units.si_to_deck_elastic` ×1e-9。源只给柔度时走 `units.py:211` `cubic_compliance_to_stiffness`
  (`pto`、`pzt06`)。
- **范围/文献值** 见 E.3 表(11 套用 `C11` 命名,`bfoP`/`btoP`/`stoP` 用旧名 `c1111/c1122/c1212`)。
- **怎么选** **C 在全仓库都是均匀的**(SPEC §0:L1 范围 homogeneous C);唯一例外是 2D 弹性 SPM
  (`elastic.py:525` `setup_elastic_spm`,WP10/§16),它接受 (Ny,Nx) 场并取**算术平均**作 C₀ 参考。
- **陷阱** 甲板给 C 不等于算例要开弹性:`elastic_params=None` 时弹性整体关闭;`nnbm`、`pbzoT`
  的 C/Q 是**存档但不用**的("parked, elastic OFF")。

#### `Q11_phys`,`Q12_phys`,`Q44_phys` — 电致伸缩系数
- **物理含义** 应变对 P² 的响应系数。Q₁₁>0、Q₁₂<0 是钙钛矿的常态:沿极化方向伸长、垂直方向
  收缩(体积近似守恒)。Q₄₄ 管剪切,直接决定 90°/109°/71° 畴壁的弹性代价。
- **进入的公式** SPEC §1.4;`elastic.py:93` `eigenstrain`、`:112` `elastic_force_explicit`。
- **单位** `m⁴/C²`,**SI == deck,不换算**(SPEC §9.1(a);应变无量纲、P 不缩放)。
- **范围/文献值** 见 E.3 表。
- **怎么选** 与 C 成套取自同一甲板,不跨甲板混搭(Q–C 的**配对**有历史,SPEC §37.5 记有一处
  Q₄₄ 配对修正待办)。
- **陷阱(§37.5 命名陷阱,必读)** 仓库有**两种 Q 约定**:`elastic.py:93`(2D,Voigt 风味)
  `e[0,1]=0.5*Q44*P1*P2`,与 `bulk3d.py:390` `eigenstrain_3d`(3D,全张量)`eps0_12=2*Q1212*P1*P2`。
  两者一致**当且仅当 Q₄₄ = 4·Q₁₂₁₂**;把张量记法的 Q₁₂₁₂ 当 Q₄₄ 用,剪切本征应变**差 4 倍**。

#### Q 的三套命名与被禁的 /4 hack
甲板里同时存在 **`Q*_phys`**(11 套)、**`Q*_voigt`**(`btoP`/`stoP`)、**`Q*_tensor`**(`bfoP`)
三种键名(`deck_parser.py:128-155` 只认 `Q*_phys` 进 R1 扇区),且 **`pzt06` 的 Q₄₄ 带着源自身
显式的 ½ 前因子**(按 SPEC §9.1(d) "keep it and note it" 原样保留,**未**对称化)。SPEC §1.4 另立
禁令:"**Never** use the legacy `Q11 Q12 Q44` material block (it carries a /4 hack for an old
kernel)"——旧甲板里那个同名块不是物理量,是给老内核的补丁,**任何情况下不得使用**。
**普查 T6 #22 判定:三套命名并存 + pzt06 的 ½ 是"活的不一致",不是一个已定的范围。**

#### `k0` — 均匀应变模式
- **物理含义** 宏观(k=0)应变 ε̄ 怎么定:`'stress_free'` = 自由块体(ε̄ 由 σ̄=0 决定);另一支是
  **受夹持/失配**模式(ε̄ 由衬底给定)。这是"块体 vs 薄膜"的物理分水岭。
- **进入的公式 / 单位** SPEC §1.4 (a)/(b);`casekit.py:95` `build_3d(..., k0='stress_free')`、
  `bulk3d.setup_elastic_3d(k0=…)`,默认 `'stress_free'`;无量纲(模式标签)。
- **怎么选 / 陷阱** 想做应变工程却忘了切模式是最沉默的一类错误:结果看着收敛且合理,只是
  **物理边界条件根本不是你以为的那个**。

#### `eps_m` — 二维失配应变
- **物理含义** 模式 (b):ε̄₁₁=ε̄₂₂=ε_m,面外 ε̄₃₃ 由 σ₃₃=0 解出。ε_m>0 拉伸(偏好面内极化),
  ε_m<0 压缩(偏好面外极化)。
- **进入的公式 / 单位** SPEC §1.4(b)/§6.2c;`elastic.py:181` `setup_elastic(..., eps_m=None)`、
  `:210` `homogeneous_strain_misfit`;默认 `None` = 自由模式;无量纲应变。
- **范围 / 怎么选** **仓库内无来源**(WP19 失配序列按算例自定)。
- **陷阱** `film_full=True` 时 `eps_m` **改口作 e₀ 用**(`elastic.py:222`
  `homogeneous_strain_film_full`——同一形参、两种语义)。

#### `e0` — 薄膜衬底失配(Li 2002 夹持形式)
- **物理含义** 衬底完全夹持时的均匀应变 **ε̄ = diag(e₀, e₀, −2(C₁₂/C₁₁)e₀)**:面内被衬底锁死,
  面外按泊松效应自由弛豫。
- **进入的公式 / 单位** SPEC WP6 §12.1/§13.2;`film_elastic.py:128` `homogeneous_strain_film_e0`
  (逐行核对 `ebar[2,2] = -(C12/C11)*2.0*e0`)、`:301` `setup_film_elastic(..., e0)`;无量纲应变,
  **必填无默认**。
- **范围/文献值** 形式本身 `[std-ref 未复核]` Li, Hu, Liu & Chen, Acta Mater **50** (2002) 395
  (SPEC §13 第 536 行引用);**具体取值按算例,仓库内无推荐范围。**

#### `hf`(膜厚)、`hs_dx`(衬底厚) — 薄膜堆栈
- **物理含义** 膜厚决定退极化能与弹性弛豫的竞争(薄 → 退极化主导);衬底厚度决定夹持有多"硬"。
- **进入的公式 / 单位** SPEC §13.0;`film3d.py:194` `make_grid_3d(Nx,Ny,Lx,Ly,Nz,hf)`(nm)、
  `film_elastic.py:98` `make_stack_grid(film_grid, hs_dx=PAPER_HS_DX)`(dx 个数,整数)。
- **值 / 怎么选** `PAPER_HF_DX = 20`(`film3d.py:190`,论文原文 "hf = 20·dx")、`PAPER_NZ = 21`
  节点(`:191`)、`PAPER_HS_DX = 12`(`film_elastic.py:89`,SPEC §13.0 引论文 "hs=12dx")
  `[LIT]`/`[toml-meta]`。
- **陷阱** `film3d.make_grid_3d` 与 `bulk3d.make_grid_3d` **同名不同签名**(前者第 6 参数是膜厚
  `hf`,后者是 `Lz`),普查标为 NAME COLLISION。

### E.2 什么时候开弹性
`elastic_params=None`(默认)= 弹性关闭(`solver.py:480/:496/:550` 直接跳过);
`casekit.build_3d(..., elastic=True)` 才建 `el`。**开弹性每步多 12 次变换**(6 次正变换 + 6 次逆变换;WP28 P6 之后的计数,SPEC §39.1 A2'(iii))。

### E.3 材料表(弹性与电致伸缩)→ §K

`C11/C12/C44`、`Q11/Q12/Q44`、每套甲板实际用的**键名变体**(`C11` vs `c1111`;
`Q*_phys` vs `Q*_voigt` vs `Q*_tensor`),以及 `pzt06` 的 ½ 前因子注,由脚本生成:
**见 §K 表 K.6(弹性)与表 K.7(电致伸缩)**。κ 的逐甲板值见表 K.5。

**缺口(T6 #6/#7/#23):`pto` 与 `pzt06` 的 κ、`pztG`/`pztH` 的 C 与 Q、Yang 2025 自身的弹性
协议 —— 仓库内无来源。** 前两者在各自 `[meta].note` 里标为 PLACEHOLDER REQUIRED。

---

## §F 动力学与数值参数

### F.0 物理与总纲

动力学是 TDGL(Allen–Cahn,非守恒):**∂P_i/∂t = −L·δF/δP_i**(SPEC §1)。这一节的参数分两类,
必须分清:**`L` 和 `kT` 是物理量**(前者定义时间单位,后者定义温度);**其余全是数值参数**——
它们只能影响"算得准不准、快不快",**不允许影响答案**。凡是改了数值参数就改变结论的,是 bug。

#### `L` — 迁移率(TDGL 动力学系数)
- **物理含义** 弛豫速率常数,把驱动力 δF/δP 换成速度。它**只定标时间**,不改变任何静态解。
- **进入的公式 / 单位** SPEC §1、§0;每个 stepper 的形参、`casekit.anneal(L=1.0)`;deck 迁移率。
- **值 / 怎么选** **恒为 1.0**,全仓库,`[DIAL §0]`(SPEC §0 原文 "Mobility L=1 (deck
  `TimeDerivativeScaled time_scale=1.0`)";§32.4 R3 "L absorbed into the time unit";`nnbm` 的
  `[meta].L_mobility` 重述了这一条)。**不要改它**——改 L 等于改时间单位。
- **陷阱 / 缺口(T6 #11)** **deck 时间因此没有 SI 锚**:仓库里没有任何测量过的动力学系数,deck
  时间**无法换算成秒**,`dt` 也**没有物理范围**,只有数值验证出来的范围。**凡需真实时间尺度
  (频率、扫场速率的绝对值)的结论,本引擎给不出。**

### F.1 时间步 `dt`:一个探测器 + 三条法则

**`probe_stable_dt` 不是收敛性检验,是有界性探测器。** `solver.py:643` 的 docstring 很直白:跑
`n_probe` 步,要求 F 有限、max|P| 不越过物理上限,**加倍**直到失败,返回**最大稳定值的一半**
(SPEC §2 的 Δt 协议);`p_cap` 自陈 "a blow-up detector, not a physics bound"。**探测通过 ≠
结果收敛**——它只保证没炸。真正的选步入口是 **`timestep.py:171` `verified_dt`**,SPEC §36.3(2C)
给了它三条法则:

1. **引擎无关**:同一函数服务 `solver`/`bulk3d`/`film3d`(以及委托给它们的 `grad4`)。
2. **固定总物理时间**(SPEC §28.6(a)):dt 与 dt/2 必须跑**同样长的物理时间**而非同样多的步数——
   固定步数的对比会给出**假收敛**(两条轨迹都还没离开初态)。
3. **探测必须带生产驱动项**(SPEC §32.5,扩展 §27.2(b)/§28.6(a)):淬火场、`E_ext` 一个都不能少;
   在案事故里 E_ext=0 的探测把稳定 dt 高估 **64×**,生产跑 NaN。

| 参数 | 位置 | 含义 | 缺省 | 源级 |
|---|---|---|---|---|
| `dt` | `solver.py:510` `build_Minv`、`:550` `step`、`:670` `relax`;`casekit.py:122` | M(k)=I+Δt·L·[A+D/κ](SPEC §2) | `None` → 走探测 | 数值 |
| `dt0` | `solver.py:643`、`timestep.py:171`、`film3d.py:537` | 探测起点 | `0.02` | `[DIAL §2]`("start 0.02") |
| `n_probe` | 同上 | 每档试跑步数 | `200` | `[DIAL §2]`("run 200 steps") |
| `growth` | 同上 | 加倍因子 | `2.0` | `[DIAL §2]` |
| `max_doublings` | 同上 | 阶梯上限 | `30` | `[DIAL]`(代码自定) |
| `p_cap` | 同上 | max\|P\| 爆炸阈 | `None` → `max(10·max\|P0\|,5.0)` | `[DIAL §2]` |
| `T_total` | `timestep.py:171` | dt/dt/2 对比的**固定总物理时间** | **必填** | `[DIAL §28.6(a)]` |
| `tol` | `timestep.py:171`;`energy_protocol.py:167` | dt vs dt/2 的相对一致度 | `0.01` | `[DIAL §28.6(a)]` |
| `max_halvings` | 同上 | 减半阶梯上限 | `10` | `[DIAL]` |
| `eps` | `timestep.py:171` | 相对误差分母下限 | `1e-12` | 数值 |
| `skip_probe`,`run_fn` | `timestep.py:171` | 逃生口(自带 dt / 自带 runner) | `False`,`None` | 数值 |
| `n1_probe`,`n_dwell_probe` | `energy_protocol.py:167` | 用缩短的回线当探测工况 | `8`,`200` | `[DIAL §29.3]` |

#### `scheme` — 时间格式
`'sbdf1'`(缺省;SPEC §2 一阶半隐式,Chen & Shen, CPC **108**, 147 (1998) `[pdf]`)或 `'sbdf2'`
(`solver.py:595` `step_sbdf2`,WP4 §10c,首步用 sbdf1 引导)。二阶只影响时间精度与可用 dt,
**不改稳态**;`solver.py:643`/`:670`、`film3d.py:537` 都收它。

#### 收敛与判停(确定性路线)

| 参数 | 位置 | 含义 | 缺省 | 源级 |
|---|---|---|---|---|
| `n_max` | `solver.py:670` `relax`;`casekit.anneal` 必填 | 步数上限 | `20000` | `[DIAL §2]` |
| `tol_F` | `solver.py:670`;`film3d.py:610` | \|dF/dt\| 判停阈 | `1e-8` | `[DIAL]`(SPEC §2 "default 1e-8 in deck units") |
| `n_plateau` | `solver.py:670` | 连续满足次数 | `20` | `[DIAL §2]` |
| `check_every` | `solver.py:670` | 每多少步算一次 F | `10` | `[DIAL §2]` |
| `tol_P` | `solver.py:670` | 可选的 max\|ΔP\| 附加判据 | `None` | `[DIAL]` |
| `dt_adapt`/`dt_grow`/`dt_cut`/`dt_min_frac`/`dt_max_frac` | `solver.py:670` | 自适应 dt(默认关,关时与旧行为逐位相同) | `False`/`1.2`/`2.0`/`0.001`/`32.0` | `[DIAL §10]` |

**陷阱**:`tol_F` 是**总能对时间的变化率**,单位是 deck F / deck 时间——它随体系大小变化。
换网格尺寸时,原来的 `tol_F` 未必还是同样的严格程度。

### F.2 有噪声时:`kT`、`noise_cutoff_nm`、判停、种子

#### `kT` — 热噪声强度(离散 FDT)
- **物理含义** Langevin 动力学 dP = −L(δF/δP)dt + √(2LkT)dW(SPEC §18)。k 空间**线性族**用
  **精确的逐模 Ornstein–Uhlenbeck** 传播子推进(衰减 e^(−Λdt) + 与之匹配的噪声协方差),
  非线性力按显式 Euler–Maruyama 漂移项处理。"精确"指**离散**涨落耗散定理成立,不是连续极限近似。
- **进入的公式** `solver.py:1048` `build_ou_operators(ops, dt, L, kT, noise_cutoff_nm=None, grid=None)`、
  `:1104` `step_langevin`;3D 是 `bulk3d.build_ou_operators_3d`/`step_langevin_3d`,薄膜是
  `film3d_langevin.build_ou_operators_film`;`casekit.anneal(kT=None)`。
- **单位与换算(SPEC V2-04.0 "CONVENTION OF RECORD",裁决 D16——三个引擎同一条)**
  `units.py:105` **`kT_langevin(T, hx_nm, hy_nm, hz_nm)` = k_B·T·1e18/(hx·hy·hz)**,是一个 **deck
  能量密度**。为什么是密度:步进器采的是 exp(−F_total/k_BT),而 F_total = Σ_cells f·V_cell
  (`total_energy`/`total_energy_3d`/film 能量自己的定义),写成模变量就是 pref·Σ_k ½χ|m|²,χ 是
  "每 P² 的能量密度",所以乘在它上头的 FDT 系数也必须是能量密度——每个**元胞体积**一份 k_B·T;
  未归一 rfft 模的稳态律因此是 ⟨|m_k|²⟩ = N_cells·kT/χ_k(步进器不再除 √N_cells,film 从来没除过)。
  **2D 的 hz 是涨落层厚 Dz,而且是个真参数**:一个**明写的建模选择**,惯例 `Dz = hx`(沿用 WP9
  `formula_units_per_cell` 的裁决),`[DIAL §18]`,**不是文献数**,现在写在每一个调用点上——因为在
  这条约定下 2D 步进器采的是 exp(−F_total·Dz/k_BT)(厚度 Dz 那层薄片的物理玻尔兹曼权重),每个 2D
  涨落方差 **∝ 1/Dz**,**换 Dz 就换掉了 2D 的热学结果**,`Dz = hx` 是个选择不是事实。("Dz 恒等
  消掉"只对**旧**约定的 T_equivalent 比值成立:错的和对的 kT 都带同一个 1/Dz,见
  `REVIEW_v2_03c_kt.md` derivation 7。)k_B = 1.380649e-23 J/K
  `[std-ref 未复核]`(2019 SI)。`units.py:162` `si_to_deck_kT` 与 `bulk3d.py:1291` `kT_deck_3d`
  算术不变,但只是"k_BT → F_total 单位换算",**不是** FDT 系数,已退出全部朗之万调用点。
- **怎么选** 由目标温度 T 定,三个引擎一律用 `units.kT_langevin`。缺省 `None` = 确定性,不加噪。
  **权威是 K5 `kt_convention`**(绝对 SI 腿 |T_equiv/T − 1| ≤ 5 %、定盒子的网格不变性腿、
  解耦双势阱的非线性绝对腿);**不要**拿 `curie_weiss_clean_control` 这类 χ = Var/kT 的观测量当
  温度锚——kT 在里面恒等消掉,它是甲板检查不是温度检查。
- **陷阱** ① V2-04.0 之前 2D 与 bulk3d 的噪声低了 N_cells/V_cell 倍(标称 T 实跑
  T·V_cell/N_cells,**没有连续极限**);Part I 按 D7 冻结并逐案重标,影响见
  `docs/CAPABILITIES.md` §6.5/§6.6 与 `reports/REVIEW_v2_03c_kt.md`。② 改对之后 0.4 nm 元胞上的
  kT 约是 bto 势阱深度的 27 倍——元胞尺度涨落的物理事实,要管住高 k 端就用 `noise_cutoff_nm`。

#### `noise_cutoff_nm` — 噪声截断长度 ℓ_c(SPEC V2-04.0,可选,缺省 `None`)
- **物理含义** 把 Hohenberg–Halperin 的截断 Λ 明写出来:|k| ≥ 2π/ℓ_c 的模,**噪声幅度**置零、
  **衰减不动**。保留下来的模严格是 T 上的玻尔兹曼分布,被排除的模确定性弛豫。
- **进入的公式** 三个 OU 构造器的关键字参数:`solver.build_ou_operators(..., noise_cutoff_nm=, grid=)`、
  `bulk3d.build_ou_operators_3d(..., noise_cutoff_nm=, grid=)`(2D/3D 都必须同时给 `grid=`,波矢在
  grid 里不在 ops 里)、`film3d_langevin.build_ou_operators_film(..., noise_cutoff_nm=)`(薄膜的
  kz,eff,j² = (μ_j − μ_0)/G11 由它自己的 z 本征问题给,均匀 z 模 kz,eff = 0)。步进函数完全不变,
  整个截断只活在 `noise_amp` 里。
- **单位** nm(deck 长度),和 `hx`/`hy`/`hz` 同一把尺。`None` = 完整的离散 FDT,返回的 OU 字典与
  不给这个参数时逐位相同。
- **怎么选** 它的用处是**收敛**:盒子与 ℓ_c 固定时,只要 h < ℓ_c/2,保留的模集合就与网格无关,
  热学观测量才会随加密收敛。所以 ℓ_c 要明显大于最细的网格间距、又小于关心的物理长度。
  `[DIAL V2-04.0]`——仓库自定的旋钮,无外部来源。
- **陷阱** 2D/3D 忘了给 `grid=` 会当场报错(`ops` 里没有波矢);被截掉的模不是"噪声变弱"而是
  **完全无噪**,所以 ℓ_c 选得太大等于在做一个更小的有效体系。

#### `T`(开尔文)
Landau 的 α₁(T) 与 kT 的共同输入,逐甲板记在 `[meta].T`(**多数 298.0**);`materials.get(name,
T=…)` 让生成式重算。**唯一以摄氏度为入口的函数**是 `film3d.py:233` `paper_alpha(T_C)`。

#### 判停:噪声下 dF 判据作废
SPEC §18 原文 "dF-plateau criteria are **VOID**"。取而代之的是**分块平稳性**(`solver.py:1087`
`block_stationary`):最近 `n_blocks_check` 个块均值相互一致到 `tol`(相对自身量级)。这是
**极差/均值检验**不是斜率拟合——随机轨迹的块均值平稳后**围绕**常数涨落,不会平滑趋近。

| 参数 | 位置 | 缺省 | 说明 |
|---|---|---|---|
| `sample_every` | `casekit.py:122` `anneal` | **必填** | 每多少步采一次总能 |
| `tol` / `n_blocks` | `casekit.py:122` `anneal` | **0.01** / 4 | `[DIAL §18]` |
| `n_blocks_check` / `tol` | `solver.py:1087` `block_stationary` | 4 / **0.1** | `[DIAL §18]` |

**两个 `tol` 缺省不一致(普查 T5 点名要求讲清谁管用):`anneal` 把自己的 `tol`/`n_blocks` 原样
转发(`casekit.py:174-175` `block_stationary(trace, n_blocks_check=n_blocks, tol=tol)`),所以
**走 `anneal` 时生效的是 0.01(更严);`block_stationary` 自己的 0.1 只在直接调用它时生效**。
另:`anneal` 在 `len(trace) >= 2*n_blocks` 前不检验(暖机),非有限态只在采样点查,故最多可能有
`sample_every−1` 步 NaN 先于报告。

#### `seed`
所有 RNG 入口(`solver.py:834` `random_smooth_field`、`casekit.anneal`、每个场生成器)缺省 `0`
`[DIAL]`。(seed, step) → 逐位可复现(SPEC §18 G16e)。**系综必须自己声明种子表**——
普查 T6 #17:**"一次无序平均需要多少个种子"这个问题,仓库内无来源**,没有任何收敛性研究。

### F.3 网格:`Nx/Ny/Nz`、`Lx/Ly`、`h`

- **物理含义** `h` 是唯一真正带物理的数值参数:必须**解析畴壁宽度 ξ ∝ √(G/|α₁|)**(§C.0)。
  h 太大 → 畴壁被网格钉住(晶格钉扎假象、错误的矫顽场);h 太小 → 白费算力且 dt 被迫变小。
  `Nx·h = Lx` 决定盒子装不装得下目标图案(条纹周期 λ\*、关联长度 ξ_disorder)。
- **进入的公式 / 单位** `solver.py:246` `make_grid(Nx,Ny,Lx,Ly)`(`hx=Lx/Nx`,`Volume=Lx*Ly` 是
  **面积**);`casekit.py:82` `build_2d(deck,Nx,Ny,h)` 取 `Lx=Nx*h`;3D `bulk3d.py:134` /
  `casekit.py:95`;薄膜 `film3d.py:194`/`:215`。`Nx/Ny/Nz` 是计数,`Lx/Ly/h` 是 nm。
- **在案的两个 h**:**dx = 2 nm**(`nnbm`,`[LIT]` Wang NC **16**, 2892 (2025) Methods,配
  128×128 周期盒)与 **dx = 0.4 nm**(`film3d.py:189` `PAPER_DX_NM`,`[DIAL §13.0]` "WP6 Tier 2's
  choice, reused verbatim",由 SPEC §13.0 的 Δx = √(G110/α₀) 定——`btoLp` 的 G₁₁₀ 正是由它**反推**
  的)。`energy_protocol.py:45` `build_case` 缺省 `delta_nm=1.0` `[DIAL]`;薄膜 `PAPER_NXY=128`、
  `PAPER_NZ=21`。
- **怎么选** 先算 ξ,让 h 明显小于 ξ,再确认 `Nx*h` 装得下目标图案。**网格是收敛性参数,换网格
  必须重跑 `verified_dt`**(dt 上限随 h 变)。
- **陷阱** 2D 的 `Volume=Lx*Ly` 是面积,所有 2D "总能"都是**单位 z 长度上的能量**(§A.1);
  这也是 kT 依赖 hx 的根源。**缺口:`Nx/Ny/Lx/Ly` 本身没有文献范围**,是逐算例的设计选择
  (`nnbm` 的 128×128 是唯一有出处的一组 `[LIT]`)。

#### `amplitude`,`kmax_frac` — 随机初态
`solver.py:834` `random_smooth_field(grid, n_comp, amplitude, kmax_frac=0.3, seed=0)`:
n_comp 个独立实场,带限到 |k| < `kmax_frac`·max|k|,再各自重标到给定 std(=`amplitude`)。
`kmax_frac=0.3` 是 `[DIAL]`(代码缺省;`1.0` = 全带宽)。**物理含义**:种子噪声要足够小以免预设
结构,又要足够平滑以免被网格尺度的高 k 分量主导——带限就是干这个的。**范围:仓库内无来源**,
两者都是逐算例的数值选择;`energy_protocol.py:25` `YANG_NOISE_AMPLITUDE_SI = 0.001` 是唯一有
协议出处的幅度 `[DIAL §27.3(d)/§29.1(ii)]`。

---

## §G 无序与淬火场（`solvers/quenched_fields.py`、`disorder.py`、`composition.py`、`composite.py`、`pzt_field.py`）

**物理前言（≤15 行）**。本仓库把「静态无序」拆成三条互不串线的机制，模块 docstring 与 SPEC §36.3(2B) 把这条边界写成结构性规定：

1. **电荷失配 → 屏蔽库仑 → 淬火随机电场 `E_d(r)`**。异价 B 位取代（PMN 的 Mg²⁺/Nb⁵⁺）让每个格点带一个相对平均价的电荷偏离。无关联点电荷过程的电荷密度功率谱由 Campbell 定理给出 `S_ρ0 = e²·Var[Δq]·n_B`（`n_B = a⁻³`，一个格点一个式量单位）；再经 Thomas–Fermi/Yukawa 型屏蔽传递函数（屏蔽长度 `λ_s`）与紫外截断 `k_max = π/a`，闭式积出 `⟨|E|²⟩`。**振幅是推导出来的，不是旋钮**（SPEC §21.1「both are outputs of the derivation, not knobs」）。这条产物只喂 `E_ext` 通道，**永远不进 alpha**。
2. **键序 / 化学有序区（COR）→ 局域 `T0` 位移 → `alpha1(r)` 数组**。SPEC §21.3 裁定 1 把电中性「random site」图像定为默认：COR 对 `E_d` 的贡献恰为零，其淬火作用改由局域 Curie–Weiss 温度移位承担。这条产物只喂 Landau 系数场，**永远不进 `E_ext`**。
3. **成分 / 标签场 → 整套系数场 + Vegard 本征应变**。`x(r)` 的随机混合极限方差 → 材料系数场（Landau/Q/C）与化学应变；二值/K 值标签场 `L(r)` 走同一条「系数按位置取值」通道。
4. 三条机制共用一个**声明式旋钮生成器** `gaussian_random_field`（`quenched_fields.py:132`）：每个 DERIVED 函数算出自己的 rms/ξ 之后**只调它一次**；想要一个赤裸的、未定向的场（W 通道的 `E_rand`）就直接调它。

### G.1 取代型点电荷无序 → `charge_disorder_stats` / `charge_disorder_field`

**`conc`** ｜ c ｜ 分数，无量纲（deck = SI，无换算）
- 物理含义：两种共存物种中**第一种**占据某个格点的概率；与 `delta_q` 一起进入两点（Bernoulli）分布方差恒等式。
- 进入的公式：SPEC §21.1；`quenched_fields.py:258` `var_dq = conc*(1-conc)*delta_q**2` → `:267` `E2 = S_rho0/eps**2 * I/(2π²)`。
- 典型范围与文献值：`PRESET_PMN.conc_substitutional = 1/3` `[toml-meta]`（PMN B 位 Mg²⁺ 概率 1/3、Nb⁵⁺ 概率 2/3，1:2 化学计量）。其他材料：仓库内无来源。
- 怎么选：由**化学计量决定**，不是拟合量。`quenched_fields.py:228` docstring 给出自洽检查：`(1/3)(2/3)(3²)=2.0` 恰好复现 `pmn_field.VAR_DQ = 2.0`。
- 陷阱：`conc` 与 `1-conc` 对方差**对称**（`c(1-c)`），所以填错哪一种物种不会报错、结果也不变；但配对的 `delta_q` 必须是**同一对物种之间的差**，否则整条链静默偏移。

**`delta_q`** ｜ Δq ｜ 电子电荷数（`E_CHARGE_SI = 1.602176634e-19` C，`quenched_fields.py:122`）
- 物理含义：两物种之间的**电荷失配**（不是相对某个第三参考的绝对偏离 —— docstring 明写这一点），是散粒噪声源的强度。
- 进入的公式：同上 `:258`；`:263` `n_B = 1/a³`，`S_rho0 = e²·var_dq·n_B`。
- 典型范围与文献值：`PRESET_PMN.delta_q_substitutional = -3.0`（`quenched_fields.py:404`；Mg²⁺ 偏离 −2e、Nb⁵⁺ 偏离 +1e，差 = −3）—— 这是**从价态推出来的**，deck 里没有这个数；deck 记的是它的后果：`pmn_disorder_R3.toml` 的 `[charge_model]` 注释写着那两个偏离，`var_dq = 2.0` `[toml-meta]`。其他材料：仓库内无来源。
- 怎么选：从价态表算，不调。电荷分布的**均值从不进入推导**（Campbell 定理只依赖方差），所以「平均电中性」不是这条链的前提。
- 陷阱：单位是 **e 的倍数**，不是库仑；把 SI 电荷塞进来会让 `E_d` 大 19 个数量级而不报错。

**`lambda_s_nm`** ｜ λ_s ｜ nm（SI = m，代码内 `lam_m = λ_s×1e-9`，无 `units.py` 函数）
- 物理含义：Thomas–Fermi 型屏蔽长度；它是白噪声源 + Yukawa 核问题里**唯一的长度标度**，因此同时也是输出场 `E_d(r)` 的关联长度（`:270` 直接返回 `lambda_s_nm` 作为 `xi_ed_nm`）。
- 进入的公式：SPEC §21.1；`quenched_fields.py:171` `_shot_noise_k_integral(kmax, κ)`，`κ = 1/λ_s`，`I = ∫₀^kmax k⁴/(k²+κ²)² dk` 闭式（import 时对数值积分自检 ≤1e-8）。
- 典型范围与文献值：PMN `default_nm = 2.0`，`range_nm = [1.0, 5.0]` `[toml-meta]`，出处 Westphal/Kleemann/Glinchuk, PRL 68, 847 (1992) + Glinchuk & Farhi, JPCM 8, 6985 (1996) `[std-ref 未复核]`。该 `[meta].note` 同时记录：**两篇论文都没有给出 PMN 的显式屏蔽长度数字**，这个区间是锚在 PMN 自身的短程结构标度（COR 2–3 nm、极性纳米区关联长度 ~1.5–7 nm）上的参数化。
- 怎么选：按 SPEC 自己的 fallback —— 用区间扫描 + 敏感性说明，不要报单点。极限自检已在 `:164-169` 注释里：`κ→0` 给 `I→kmax`（方差被晶格截断主导），`κ→∞` 给 `I→kmax³/(3κ²)→0`（完美屏蔽杀死全场）。
- 陷阱：λ_s 同时控制**振幅和关联长度**，不能当成两个独立旋钮；改 λ_s 会同时移动 G18a 的两个验收量。

**`eps_r`** ｜ ε_r ｜ 无量纲（`EPS0_SI = 8.8541878128e-12` F/m，`quenched_fields.py:123`）
- 物理含义：屏蔽**局域电荷–电荷相互作用**的背景/光学介电常数，`ε = ε₀ε_r` 出现在 `E_d` 振幅的分母平方上。
- 进入的公式：`quenched_fields.py:267` `E2 = S_rho0/eps**2 * I/(2π²)`。
- 典型范围与文献值：PMN `bracket = [500, 1000, 2000]`，`default = 1000.0` `[toml-meta]`。**这是经验排除法得到的括号，不是测量值** —— 原始 (10,50,100)/默认 50 的括号被 SPEC §22（WP16/R4 裁定 2）**经验排除**：PMN 实验上在 Burns 温度以上呈 Curie–Weiss 行为，而 ε_r=50 的参数化在任何可信温度都没有可达的 Curie–Weiss 窗口。**范围：仓库内无 PMN 专属的背景介电常数来源**（T6 #10）。
- 怎么选：只用 `[screening_eps_r].bracket` 里的三个值，并在报告里写明它是排除法结论。
- 陷阱：**不要**用弛豫体巨大的低频介电常数（~20000–30000）—— `[meta].note` 明写那本身就是被建模的软模响应，代入会构成循环论证。

**`a_lattice_nm`** ｜ a ｜ nm
- 物理含义：赝立方钙钛矿晶格常数，给出格点密度 `n_B = a⁻³` 与紫外截断 `k_max = π/a`（「一个晶格间距是连续介质电荷密度还有物理意义的最细尺度」）。
- 进入的公式：`quenched_fields.py:262-263`。
- 典型范围与文献值：PMN `0.4049` nm `[toml-meta]`（`[a_lattice].note`：标准晶体学值）；PZT `A_LATTICE_PZT_NM = 0.406` nm（`pzt_field.py:89`，与 Vegard 斜率同源）`[std-ref 未复核]`。
- 怎么选：查该材料自己的赝立方点阵常数，不与网格间距 `h` 混用。
- 陷阱：`a` 同时进 `n_B`（∝a⁻³，抬振幅）和 `k_max`（∝a⁻¹，也抬振幅），敏感度比直觉高；它**不是**网格分辨率参数。

**`seed`**（本节所有生成器共用）｜ — ｜ 整数
- 物理含义：淬火构型的随机实现标签 —— 不是物理量，而是「这块样品」的编号。
- 进入的公式：`quenched_fields.py:132` `gaussian_random_field(grid, sigma, xi_nm, seed)`；矢量场的三个笛卡尔分量**各自独立**用 `seed+i`（`:280`、`:346`、`pzt_field.py:266`）。
- 典型范围与文献值：默认 `0`（代码默认）`[DIAL]`。**系综需要几个种子才收敛：仓库内无来源**（T6 #17 —— 唯一存在的 `n_seeds=14` 出自 `susceptibility.calibrate_fl_chi_bias`，用途完全不同）。
- 怎么选：把种子集合**显式写进算例并记进报告**；跨机制比较时对齐种子，只有这样差分才干净。
- 陷阱：三分量用 `seed+i`，所以「场 A 用 seed=0、场 B 用 seed=1」会让 A 的第 2 分量与 B 的第 1 分量**完全相同**。分配种子时按 3 递增。

### G.2 带电纳米区无序（默认关闭）→ `region_charge_disorder_stats/_field`

**`f_region`** ｜ f ｜ 体积分数
- 物理含义：化学有序区（COR）占的体积分数；与 `delta_q_region` 一起经两点分布给出区域/基体电荷涨落的 rms。
- 进入的公式：SPEC §21.1 第 2 条；`quenched_fields.py:301`、`:291` `_region_matrix_charge`（由全局电中性反解基体电荷 `f·Δq_COR + (1-f)·Δq_matrix = 0`）、`:333` `E2 = S0/eps²·I/(2π²)`。
- 典型范围与文献值：`0.20` `[toml-meta]`，出处 Davies & Akbas, J. Phys. Chem. Solids 61 (2000) 159–166, p.160 `[std-ref 未复核]`，`[cor_layer].quote` 逐字记录了「~20 vol% of 2-3 nm 1:1 ordered domains」。
- 怎么选：与 `xi_region_nm` 成对取自同一篇文献的同一句话，不要混搭。
- 陷阱：这一层按 SPEC §21.3 **默认关闭**（`[cor_layer_legacy_space_charge].status = "DEFAULT OFF"`）；它是被标注的历史/敏感性路径，默认路径是 G.3 的键序模型。

**`xi_region_nm`** ｜ ξ_COR ｜ nm
- 物理含义：COR 的关联长度（有限尺寸源），使源谱变成 `S_ρ(k) = S0·exp(-k²ξ²/4)`。
- 进入的公式：`quenched_fields.py:200` `_correlated_source_k_integral`（无闭式，用 `u = k·ξ` 无量纲替换后数值积分）；输出关联长度 `:336` `xi_ed = sqrt(ξ_region² + λ_s²)`。
- 典型范围与文献值：`xi_cor_default_nm = 2.5`，`xi_cor_range_nm = [2.0, 3.0]` `[toml-meta]`，同一句 Davies & Akbas 引文。**2.5 是被声明的区间中点，不是测量值** `[DIAL §21.3]`（T6 #13）。
- 怎么选：报 2.0/2.5/3.0 三点，别只报中点。
- 陷阱：`:200` 注释记录了一个真实事故 —— 未做 `u = kξ` 替换时，原始 SI 波数（~1e8–1e9 rad/m）会让 `scipy.integrate.quad` **静默返回小约 1e14 倍的错误值且不抛异常**。任何仿写这个积分的新代码都必须保留替换。

**`delta_q_region`** ｜ Δq_COR ｜ 电子电荷数
- 物理含义：空间电荷图像下 COR 每个 B 位的净电荷（1:1 Mg:Nb 化学的 −1/2，由富 Nb 基体补偿）。
- 进入的公式：`quenched_fields.py:301`/`:340`，经 `:291` 的电中性约束。
- 典型范围与文献值：`-0.5` `[toml-meta]`（`pmnD [charge_model].dq_cor = -0.5`；派生输出 `layer2_std_component_deck = 6.590148e-01`、`layer2_xi_ed_nm = 3.2016`，均为 `[toml-meta]` 记录的默认参数下推导结果）。
- 怎么选：与 `f_region` 同源；此层默认关闭，只在敏感性研究里打开。
- 陷阱：SPEC §21.1 原文写的是空间电荷图像，SPEC §21.3 已**推翻**它改用电中性 random-site 默认。读旧文本会得到相反的默认设置。

### G.3 键序 / COR → `bond_order_alpha1_field`（SPEC §21.3 默认模型）

**`T` / `T0_K` / `C_cw_K`** ｜ T, T₀, C ｜ K（Landau 输出经 `si_to_deck_landau` ×1e-9 转 deck）
- 物理含义：清洁极限的 Curie–Weiss 关系把局域转变温度映成局域 Landau 二次系数：`α₁(r) = (T − T₀(r))/(2ε₀C)`。`T0_K`/`C_cw_K` 是材料**自己**的 Curie–Weiss 常数，是必填参数（没有默认值 —— 这正是这次泛化的要点：它们是化学，不是硬编码数字）。
- 进入的公式：SPEC §21.3 裁定 2（hybrid a'）；`quenched_fields.py:387` `a1_si = (T - T0_field)/(2*EPS0_SI*C_cw_K)`，`:388` `return units.si_to_deck_landau(a1_si)`。
- 典型范围与文献值：PMN `T0_K = 398.0`（`T0_range_K = [396, 400]`）、`C_CW_K = 1.2e5`（`C_CW_range_K = [1.2e5, 1.3e5]`）`[toml-meta]`；主引 Viehland, Jang, Cross & Wuttig, PRB 46, 8003 (1992) `[std-ref 未复核]`，交叉核对 Al-Barakaty et al., PRB 91, 214117 (2015)（C=1.25e5 K / T0=396 K）。默认参数下的派生值 `alpha1_deck_at_298K = -4.705871e-02` `[toml-meta]`。
- 怎么选：用该材料**自己测得的高温 Curie–Weiss 拟合**（SPEC §21.3 明确否掉了从 PMN-PT 外推）。
- 陷阱：**因子 2 已被标记**。`pmnD [pure_pmn_curie_weiss].factor_of_2_note` 记录：SPEC §21.3 转述的公式漏了本项目自己（经 G6a 门验证）的约定里的因子 2（`literature_params.pzt_haun_x_landau_si` 的 `a1=(T-T0)/(2ε₀C₀)`，即标准 Landau 结果 `a1 = 1/(2ε₀χ)`）。实现**带**因子 2，显式标记而非静默选择。拿别处的 `C` 来用之前先确认对方的因子约定。

**`dT0_region_K`** ｜ ΔT₀ ｜ K
- 物理含义：COR 内部局域 `T₀` 的移位幅度；rms 由与 G.1 同一个两点分布恒等式给出（`quenched_fields.py:153` `_bernoulli_mixture_std`，`:384` `rms_dT0 = |ΔT₀|·sqrt(f(1-f))`），再经 Curie–Weiss 映成 `α₁(r)`。
- 进入的公式：`quenched_fields.py:359` → `:384-388`。
- 典型范围与文献值：默认 `25.0` K，括号 `dT0_cor_bracket_K = [-50, 50]` `[toml-meta]`；派生 `rms_alpha1_deck_at_default = 4.705871e-03` `[toml-meta]`。**这是括号内的旋钮，不是文献值** `[DIAL §21.3]`：电中性图像抽掉了别处用的电荷杠杆，`[meta].note` 明写「no PMN-specific literature number found」。
- 怎么选：符号不定 —— 括号两端都要跑；SPEC §21.3 把它挂在「pending R4 sensitivity」上。
- 陷阱：把它当文献值引用是错的；它是本手册里**唯一**一个默认值落在自己记录的括号内部的无源幅度。

### G.4 声明式旋钮生成器与通用高斯场

**`gaussian_random_field(grid, sigma, xi_nm, seed)`** ｜ σ, ξ ｜ σ 随场而定；ξ 为 nm
- 物理含义：单分量、零均值、谱关联的高斯随机场，**精确**命中目标 RMS = σ 与关联长度 ξ。它是本仓库把「声明式旋钮」写成结构的地方（模块 docstring / SPEC §26.0(a)(b)）。
- 进入的公式：`quenched_fields.py:132`；数值层实现在 `disorder.py:65`。
- 典型范围与文献值：**按构造无文献值** `[DIAL]` —— 调用方负责先推导出自己的 σ/ξ 再调它一次。
- 怎么选：优先走 DERIVED 函数（G.1–G.3）；只有在需要一个赤裸的、已标定的场时才直接调（W 通道的 `E_rand` 是唯一在册例子）。
- 陷阱：直接调它就等于**声明这是一个旋钮**，报告里必须这样标注，不能事后包装成推导结果。

**`disorder.random_field_gaussian(grid, rms, xi, seed)`** ｜ rms, ξ ｜ 同上
- 物理含义：数值原语 —— 白噪声 → 谱高斯滤波 → 零均值 → 精确重标到目标 RMS；功率谱 `~exp(-k²ξ²/4)`，实空间关联 `~exp(-r²/ξ²)`。
- 进入的公式：SPEC §6.2b；`disorder.py:65-81`（`amp_filter = exp(-k²ξ²/8)` 是**振幅**滤波，平方后才是功率谱）。
- 典型范围与文献值：无默认值；范围由调用方的物理层决定。
- 怎么选：不要直接调参 rms —— 让上层的推导函数算出来。
- 陷阱：`xi` 的单位是**网格长度单位**（deck 的 nm），不是格点数；Δ≠1 nm 的算例里两者不同。

**`disorder_sigma_max_deck` / `disorder_xi_erand_nm`（nnbm 的 E_rand）** ｜ σ_max, ξ_Erand ｜ deck 电场 = V/m ×1e-9 ／ nm
- 物理含义：Wang 2026 NaNbO₃ 通道的高斯随机电场，经既有的 `E_ext(r)` 通道作用在局域 P 上（SPEC §32.4 R4）。
- 进入的公式：`disorder.py:65` 生成 → `solver.py` 的 `E_ext` 通道（SPEC §32）。
- 典型范围与文献值：`σ_max = 0.218` deck = **2.18 MV/cm** `[LIT]`（`reports/LIT_wang2026_nanbo3.md` (iii) 逐字端点，`[meta].disorder_sigma_source` 转录原句「from 0 to 2.18 MV/cm」）；`ξ_Erand = 25.6` nm = λ\*/2 `[DIAL §32]` —— `[meta].disorder_xi_basis` 明写「**两篇论文都没有给出关联长度**」。
- 怎么选：σ 走 {0 → 2.18 MV/cm} 的阶梯（≥4 点含 0，≥2 个种子），ξ 只能作为已声明旋钮报告。
- 陷阱：**deck 电场因子 ×1e-9 不是「纯约定」，是导出量**（V2-02.1；T6 #12 已关闭）—— 能量密度 1e9 J/m³ ÷ P 的 1 C/m² = 1e9 V/m，所以 `E_deck = E_SI×1e-9` 是唯一自洽解（§A.4 坑 3）。nnbm `[meta]` 自己那句仍是在案交叉核对：`0.10` deck ↔ 100 MV/m = 1e8 V/m，`1e8×1e-9 = 0.10` 精确吻合；独立的门是 K5 `permittivity_convention` 的 Leg C。`charge_disorder_field` 转换 `E_d` 时长期借用 `units.si_to_deck_landau`（`quenched_fields.py:279`、`:345`）—— 因子相同、命名误导，V2-02 起改指专用的 `units.si_to_deck_field`（逐位相同，不动任何数）。

**`make_random_strain(grid, eps_rms, xi, seed, components)` / `assemble_eps_star_static`** ｜ ε\*_rms, ξ ｜ 应变无量纲 ／ nm
- 物理含义：静态随机本征应变无序 —— 每个被命名的分量一个独立种子的场（`seed+i`），组装成对称 (3,3,Ny,Nx) 张量喂给 `elastic.setup_elastic(eps_star_static=...)`。
- 进入的公式：SPEC §6.2b / §1.4；`disorder.py:83`（生成）、`:91`（组装）；批量版 `:132`、母场版 `:183`。
- 典型范围与文献值：`components` 默认 `("xx","yy","xy")`（代码默认）；`eps_rms` 与 `xi`：**范围：仓库内无来源**（T6 #13）。
- 怎么选：只能作为已声明旋钮扫描；缺省分量按 0 处理（`assemble_eps_star_static` 的约定）。
- 陷阱：`make_random_strain_master` + `resample_strain_dict` 是为跨网格尺寸复用**同一块淬火样品**准备的；换网格时若重新生成而不重采样，等于换了样品，分辨率研究会被无序涨落污染。

### G.5 成分场与 Vegard 应变（`composition.py`）

**`x_bar`** ｜ x̄ ｜ 摩尔分数，无量纲
- 物理含义：随机取代型固溶体的平均成分；所有系数场（Landau/Q/C）都是 `x` 的函数。
- 进入的公式：`composition.py:115` `composition_field` 把零均值涨落加回 `x_bar`；`:201` `deck_field(x_field, material_fn, ...)` 把 `x(r)` 映成系数场。
- 典型范围与文献值：按材料体系而定；PZT 形态相界附近 x≈0.48 有在册出处（`pzt_field.py:74-85` 注释）`[std-ref 未复核]`。
- 怎么选：由算例的目标组分决定。
- 陷阱：`x_bar` 与 `std` 是**两个**参数，但 `std` 不该独立调（见下条）。

**`std`（`composition_std` / `composition_std_correlated` 的输出）** ｜ σ_x ｜ 无量纲
- 物理含义：**混合极限**方差 —— 对 `N_f` 次独立 B 位抽样求平均，方差 = `x(1-x)/N_f`。有关联版本改为对一个**关联体积**求平均：`σ = sqrt(x(1-x)·a³/(ξ_x²·t_eff))`，`t_eff` 默认取 `ξ_x`（各向同性关联体积）。
- 进入的公式：`composition.py:94`（`sqrt(x(1-x)/N_f)`）、`:103`（关联版）、`:75` `formula_units_per_cell`。
- 典型范围与文献值：由公式**推导**，无独立文献值。
- 怎么选：docstring 的法则是「**ξ_x 才是物理旋钮（工艺/团簇尺度）—— 扫它，绝不单独扫 std**」。
- 陷阱：`composition_std_correlated` 的 `a_lattice` 现在是**必填参数**（SPEC §36.3(2B) 泛化后不再默认取 PZT 值）；老代码路径里省略它会直接报错，而不是静默用 PZT 数。

**`a_lattice`** ｜ a ｜ nm — 见 G.1 同名条目；此处只用于数格点（`formula_units_per_cell`）与给关联长度设下限。
**`xi_x`** ｜ ξ_x ｜ nm
- 物理含义：成分团簇长度（工艺尺度）。
- 进入的公式：`composition.py:115` `xi_eff = max(xi_x or 0.0, a_lattice)`。
- 典型范围与文献值：**范围：仓库内无来源**（T6 #13）；默认 `None`（等价于取下限）。
- 怎么选：扫 ξ_x，其余由公式跟随。
- 陷阱：**存在一个硬下限** —— 场永远不会在小于一个晶格常数的尺度上被平滑（随机混合的 std 公式本身就假定至少对一个式量单位求了平均）。填 `xi_x = 0.1 nm` 不会报错，只会静默变成 `a_lattice`。

**`dlna_dx`** ｜ d ln a/dx ｜ 每单位 x（无量纲）
- 物理含义：Vegard 化学应变斜率 —— 局域成分偏离产生各向同性本征应变 `ε_chem = dlna_dx·(x − x̄)·I₃`。
- 进入的公式：SPEC §15.3；`composition.py:133` `vegard_eigenstrain_field(x, x_bar, dlna_dx)` → `disorder.assemble_eps_star_static` → `elastic.setup_elastic`（SPEC §1.4）。
- 典型范围与文献值：PZT `DLNA_DX_PZT = -0.040`，**不确定度 ±0.015**（`pzt_field.py:85` 及其上方注释）`[std-ref 未复核]`。该注释逐条记录：赝立方胞参数 `a_pc = (a²c)^(1/3)`，`da_pc/dx ≈ -0.16 ± 0.05 Å` 每单位 x，`a_pc(x≈0.48) ≈ 4.06 Å`；符号来自 Ti⁴⁺（0.605 Å）小于 Zr⁴⁺（0.72 Å）。出处按注释为 Shirane/Hoshino/Suzuki (1952) 数据经 Jaffe/Cook/Jaffe (1971) 谱系的二次文献**图读**（非数字化表格），细节见 `WP9_REPORT.md`。
- 怎么选：把 ±0.015 当**诚实区间**跑三点（−0.025 / −0.040 / −0.055），别报单点。
- 陷阱：它是**图读**值，不是表格转录；引用时必须带这个限定，也必须带不确定度（census T1b：不确定度只记在注释里，没进 `[meta]`）。

**`mix_decks(fractions, decks, rule='linear')`** ｜ — ｜ 分数
- 物理含义：N 元线性 deck 混合（SPEC §36.3(2B)）—— 按体积/摩尔分数对系数做线性插值。
- 进入的公式：`composition.py:148`。
- 典型范围与文献值：`rule` 默认 `'linear'`（numerical-only）；`fractions` 由算例设计决定，无文献范围。
- 怎么选：只在两端 deck 同扇区、同约定（尤其 Q 的 Voigt/tensor 命名一致）时才用。
- 陷阱：线性混合对 Landau 高阶项**没有物理保证** —— 它是插值约定，不是混合律。

### G.6 复合相标签场（`composite.py`，SPEC §30）

**`x_AN` / `xi_AN` / `seed`（`make_label_field`）** ｜ x_AN, ξ_AN ｜ 体积分数 ／ nm ／ 整数
- 物理含义：静态二值相标签场 `L(r) ∈ {0=host, 1=filler}`；由关联高斯场经 **rank-and-cut**（取最大的 `round(x_AN·N)` 个格点）生成，因此实现分数**精确**到 ±1 格。
- 进入的公式：SPEC §30.2 G24a；`composite.py:93-114`。
- 典型范围与文献值：**范围：仓库内无来源**（T6 #16，逐案设计选择）。SPEC §30.2 G24e 的最小演示用 `x_AN ∈ {0, 0.1, 0.2, 0.3}`（SPEC 声明的算例设置，非文献值）。
- 怎么选：`x_AN=0` 必须逐位复现纯 host 引擎（G24b 的零泄漏检验），把它当接线自检跑一遍再上生产。
- 陷阱：是 **rank-and-cut 而非固定阈值**，所以同一个 `seed` 在不同 `x_AN` 下给出的**不是**嵌套的区域族；比较不同 `x_AN` 时形貌会整体重排。

**`fractions` / `xi`（`make_label_field_multi`、`_multi_3d`）** ｜ — ｜ 分数 ／ nm
- 物理含义：K 元 rank-and-cut 标签场（2D `composite.py:178`，3D `:273`）。
- 进入的公式：同上。
- 典型范围与文献值：**范围：仓库内无来源**（T6 #16）。
- 怎么选：`fractions` 之和须为 1；其余同上。
- 陷阱：3D 版是**独立函数**，不是 2D 版加一维；positional 调用别串。

**`host_alpha_deck` / `filler_alpha_deck`（`alpha_field`）、`material_alpha_decks`（`alpha_field_multi`）** ｜ α_i(r) ｜ deck Landau 单位
- 物理含义：`α_i(r) = host_i·(1−L) + filler_i·L`，逐个 `ALPHA_KEYS` 键做标签加权 —— 落在 SPEC §1.1 允许「α₁ 可以是空间数组」的既有通道上。
- 进入的公式：`composite.py:116`（二值）、`:211`（K 元）→ `solver.bulk_force` / `bulk_energy_density`。
- 典型范围与文献值：取自各自 deck（见 §K 表 K.2）。
- 怎么选：梯度项是**共享的单一 G**（SPEC §30.1 明写「Gradient shared (single G, flagged)」）—— 这是被标记的近似，不是物理结论。
- 陷阱：`L≡0` 时 `host·1.0 + filler·0.0` 是浮点精确的，所以零泄漏是构造性的而非特判；但反过来，**任何**改成阈值/插值的「改进」都会破坏这条性质。

**`eps_r`（`pe_filler_landau_si`）** ｜ ε_r ｜ 无量纲
- 物理含义：PE（非极性）填料的线性介电常数；填料退化为纯线性介质 `a1 = 5.645e8/ε_r`（SI，N·m²/C²），所有高阶/离轴项为零。
- 进入的公式：SPEC §30.2；`composite.py:48`（Yang SI Table 2 的 PE 行逐字转录）；修正版 `:57`。
- 典型范围与文献值：Yang 主文「A permittivity of approximately **50** was sufficient」`[LIT]`（SPEC §30.1；SPEC §27.3(b) 已把它**正确归属给填料**）。SPEC §30.2 G24e 用 `eps_r=50`。
- 怎么选：这是复合材料的**自由旋钮**（SPEC §30.1 原话「eps_r the free dial」），按设计扫。
- 陷阱：**命名冲突警告** —— 这个 ≈50 与 `pbzoT` 的 `kappa_b=50` 是**两个完全不同的量**。`pbzo_triplewell_yang2025.toml [meta].kappa_b` 用整段文字记录了这次误引：主文里唯一的「~50」是 PE 填料的 ε_r，**不是** κ_b。见 §H 的 `kappa_b` 条目。

### G.7 PZT 成分相关系数场与缺陷场（`pzt_field.py`）

**`x` / `T`（`landau_field(x,T)`、`Q_field(x)`、`elastic_C_field(x, mechanics=True)`、`elastic_C_at`）** ｜ x, T ｜ 无量纲 ／ K
- 物理含义：Haun 谱系的 PZT 系数随成分 x 与温度 T 的闭式依赖 —— `α₁(x,T)` 经 `T₀(x)`、`C₀(x)`，`α₁₁/α₁₁₁/α₁₁₂` 经 App A 的 η₁/η₂ 公式给出 `α₁₂/α₁₂₃`；返回 deck 单位 dict，8 阶键钉为 0.0（Haun 势只到 6 阶）。
- 进入的公式：SPEC §15.1、§16；`pzt_field.py:92`（Landau）、`:115`（Q）、`:124`（弹性 C 场）、`:143`（标量点值）。
- 典型范围与文献值：Rabe/Ahn/Triscone (2007) App A (Chen) p.367 公式集 + Table 4 柔度 `[std-ref 未复核]`；deck 快照见 `pzt_Haun_x0.6`（T=298 K, x=0.6）。
- 怎么选：`x` 与 `T` 由算例的材料点决定；数组 `x` 已被向量化（`materials.pzt_haun_x_landau_si` 在 WP9 的 `_pzt_C0` 向量化后是 array-safe 的）。
- 陷阱：`pzt_Haun_x0.6` 的 **Q44 保留了源文献自己的显式 ½ 前因子**（未做对称化）`[toml-meta]`；跨 deck 比较 Q 时这是 SPEC §37.5 的 Voigt-vs-tensor 活坑（T6 #22）。另：`pzt_Haun_x0.6` **没有** `G110`、**没有** `permittivity`（T6 #1/#6）。

**`rms` / `xi` / `seed`（`defect_field`）** ｜ — ｜ deck 电场 ／ nm ／ 整数
- 物理含义：场值化的 `E_ext`，形状 (3,Ny,Nx)，由 `solver.py` 的 `_reshape_vec3` 在任何原本只接受常量三矢量的地方消费。
- 进入的公式：`pzt_field.py:262-266` → `solver.py:425/:442/:469` 的静电力（SPEC §1.3）。
- 典型范围与文献值：**范围：仓库内无来源**（T6 #13 —— `rms` 与 `xi` 都没有任何出处）`[DIAL]`。
- 怎么选：只能声明为旋钮并扫描；单位与 §G.4 的 deck 电场约定一致。
- 陷阱：它进的是 `E_ext` 而不是 alpha —— 与 G.1 同一条通道，若同时打开两者要说清是**两个**独立的淬火层。

---

## §H AFE q 扇区（`solvers/kittel_afe.py`、`solvers/loopfit.py`）

**物理前言（≤15 行）**。Kittel 双子晶格图像把反铁电写成两个反向子晶格：`P = (Pa+Pb)/2`、`q = (Pa−Pb)/2`，即 `Pa = P + q`、`Pb = P − q`（SPEC §23.1）。`q` 与 `P` 用同一套 deck 单位。两条**载荷性**约定被门测死（SPEC §23.1）：(i) **q 不产生退极化/泊松场**（它是区边界模，介观没有净偶极），所以 `build_ops_q` 只有 `Aq(k)=Gq·k²`，没有 `D(k)`；(ii) **E 只通过 f_Pq 间接耦合到 q**，没有线性 `E·q` 项 —— 这两条正是双电滞回线能出现的原因。
本仓库有**两条**AFE 路线：**(A) 精确变换 deck** `pbzo_kittel_lum2022`（SPEC §23.5）—— 把 Lum, Lim & Chew (2022) 的子晶格泛函符号变换到 (P,q) 基，`t1 = 3β < 0` 等系数被**锁死**，不是自由滑块；这条 deck 里的 AFE/FE 竞争活在变换给出的二次项劈裂上（α_P 比 α_q 硬）。**(B) 单 P 三阱工程 deck** `pbzo_triplewell_yang2025`（SPEC §27）—— 不显式带 q，用一个六次单轴势的三阱形貌产生双回线，由 `loopfit` 从测得的回线阈值反解 (a1,a11,a111)。

**`aq1` / `aq11` / `aq12` / `aq111` / `aq112` / `aq123`** ｜ α^q_n ｜ 1e9 J/m³/(C/m²)^2n（SI = J·m/C² 等）／ `si_to_deck_landau` ×1e-9
- 物理含义：`q` 自己的 Devonshire 多项式系数，与 P 扇区同构（同一套 `solver.bulk_*` 直接作用在 q 场上）。
- 进入的公式：SPEC §23.1；`kittel_afe.py:279` `_explicit_N_q` 里的 `solver.bulk_force(q, alpha_q)`，能量在 `:333` `total_energy_pq`。
- 典型范围与文献值：仅 `pbzo_kittel_lum2022` 定义：`-0.0975, -0.25, 0, 2.66667, 0, 0` `[toml-meta]`（P 扇区对应为 `0.1025, -0.25, 0, 2.66667, 0, 0`）。源：Lum, Lim & Chew, J. Phys.: Condens. Matter **34**, 415702 (2022), doi:10.1088/1361-648X/ac7e99, Table 1，**开放获取 CC-BY** `[std-ref 未复核]`（census T4 记录：Table 1 的数字只有 `[toml-meta]` 一级，`literature/` 下**没有**该 PDF，也没有 `LIT_` 抽取笔记）。SPEC §23.5 记录源参数：`α0=5e5, β=-5e8, γ=8e9, g=1e8` SI，`T0=493 K`。
- 怎么选：不选 —— 这是精确基变换的产物。α_P 与 α_q 的差（`0.1025` vs `-0.0975`）就是变换给出的二次项劈裂。
- 陷阱：G20a 的第 0 步是 **sympy 变换核对**（符号搭出源文献的子晶格泛函，变换到 (P,q)，逐系数比到 1e-12）—— 任何手改这六个数的行为都必须重跑它。

**`t1` / `t2`** ｜ t₁, t₂ ｜ 同 α₁（能量密度）／ ×1e-9
- 物理含义：四阶 P–q 交叉耦合。`f_couple` 的精确形式在 `kittel_afe.py:127-132`：`t1·Σᵢ Pᵢ²qᵢ² + t2·(P·q)² + …`。`t1 > 0` 给相互压制 → 场致 AFE→FE 一级转变；本 deck 里 `t1 = 3β < 0` 被变换锁死。
- 进入的公式：SPEC §23.1/§23.5(a)；`kittel_afe.py:140`（能量）、`:146`（δf/δP）、`:152`（δf/δq），键名表 `:134` `COUPLE_KEYS = ("t1","t2","u1","u2")`。
- 典型范围与文献值：`pbzoK`：`t1 = -1.5`、`t2 = 0` `[toml-meta]`。`t2 = 0` 是**必填的各向异性槽位**：源泛函是共线/标量的，SPEC §23.5(e) 规定各向同性矢量嵌入为默认、所有各向异性槽位默认零。
- 怎么选：精确变换 deck 里不选。SPEC §23.5(a) 明写：`t1>0` 式的通用滑块**只留给未来的非 Kittel deck**。
- 陷阱：「force」在本模块里指 `df/dP`（与 `solver.bulk_force` 的命名一致），**不是**物理的 `−δF/δP`；调用方把它加进 N 的方式与 `bulk_force` 完全平行。照抄符号会翻号。

**`u1` / `u2`** ｜ u₁, u₂ ｜ 同上
- 物理含义：六阶交叉项 `u1·Σ Pᵢ⁴qᵢ² + u2·Σ Pᵢ²qᵢ⁴`（`kittel_afe.py:130-131`），来自源文献 β、γ 的展开闭合。
- 进入的公式：同 `t1`。
- 典型范围与文献值：`pbzoK`：`u1 = u2 = 40` `[toml-meta]`。
- 怎么选：同 `t1` —— 锁死。
- 陷阱：SPEC §23.5(a) 把项菜单**扩到**了变换的闭合形式；若给新 deck 只填 t 而漏 u，泛函就不是那个变换的闭合式了。

**`Gq`** ｜ G_q ｜ 同 G₁₁（1e9 J/m³·nm²/(C/m²)²）／ `si_to_deck_gradient` ×1e9
- 物理含义：q 的各向同性梯度刚度，`f_gradq = ½·Gq·|∇q|²`（三分量求和）。
- 进入的公式：SPEC §23.1；`kittel_afe.py:165` `build_ops_q`（`Aq(k) = Gq(kx²+ky²)`，标量、三分量相同、**无** D(k) 与 G12/G44/G44p 交叉项）、`:172` 能量密度。
- 典型范围与文献值：`pbzoK`：`Gq = 1.0` `[DIAL §23.5(d)]` —— **借用 BTO 的 G110=1.0，标记为占位符**。**范围：仓库内无来源**（T6 #4：没有任何来源发表过 q 或 θ 的梯度系数）。
- 怎么选：只能声明为旋钮；SPEC §23.5(d) 明确它走的是「PZT 梯度比借用」的先例。
- 陷阱：BTO 的 `G110 = 1.0` **本身就是重整化的参考值而非测量值**（`units.py:54-58` 明写，T6 #1），所以这里是「借了一个本来就不是数据的数」—— 两级都要在报告里说清。

**`Lam11` / `Lam12` / `Lam44`** ｜ Λ₁₁, Λ₁₂, Λ₄₄ ｜ m⁴/C²，**SI == deck，无换算**（SPEC §9.1(a)）
- 物理含义：q 的电致伸缩式本征应变贡献 `ε*_ij += Λ_ijkl q_k q_l`（SPEC §23.1 的 `f_elq`）。
- 进入的公式：`kittel_afe.py:200` `with_q_eigenstrain`、`:213` `elastic_force_explicit_q`（符号自检 `:226`）。
- 典型范围与文献值：仅 `pbzoK`：`0.0203 / −0.00757155 / 0.00879853` `[toml-meta]`（`Lam12` 按 PZT x=0.4 的 Q12/Q11 比例构造，2026-11 Q12 符号修正后重算，旧值 +8.58892e-05）。各向同性锚点由 SPEC §23.5(d) 给出：`Q(PbZrO3) = 2.03e-2 m⁴/C²`，出处「Uchino 1981 via Rabe review restatement — one attribution level removed, say so in [meta]」`[std-ref 未复核]`；立方劈裂为**必填**，起点允许用 PZT 比例并标记未验证。
- 怎么选：默认可由 deck 置零（SPEC §23.1「may be zeroed by deck」）；WP20/WP17 的门测都在 elastic OFF 下跑（源泛函本身无应变）。
- 陷阱：**除 pbzoK 外没有第二个 deck 有 Λ**（T6 #14）；`Lam11 = 0.0203` 就是那个各向同性锚点本身，把它当独立测量值引用是错的（少了一级归属）。

**`kappa_b`（AFE deck 侧，值 50）** ｜ κ_b ｜ 乘子，无量纲；deck 介电率 = `κ_b × 0.08854187`（基常数本身 = 10ε₀，所以**真 ε_b = 10·κ_b**，V2-02.1）
- 物理含义：均匀背景介电常数（每个格点上加的背景项），进 D(k)/κ（SPEC §1.3）。
- 进入的公式：`energy_protocol.py:113` `kappa_for(kappa_b) = kappa_b * _KAPPA_UNIT_EPSB10`（`:49`）；deck 里 `permittivity = 4.42709`（= 50 倍那个基常数）`[toml-meta]`；该基常数的名字在 V2-02.1 由 `_EPS0_DECK_BASE` 改成 `_KAPPA_UNIT_EPSB10` —— 旧名断言它是 ε₀，那是错的，它是 10ε₀，所以该甲板真正跑的是 **ε_b = 500** `[DIAL V2-02.1]`。
- 典型范围与文献值：**范围：仓库内无来源（T6 #5）**。`pbzo_triplewell_yang2025.toml [meta].kappa_b` 逐字记录：主文的 eq.6/9 里 K 只有**符号定义**（"background dielectric constant… K"），从未给数；正确出处标注为「CONVENTION-INHERITED, UNSOURCED」`[DIAL §26.2]`。括号 `[25, 100]` 的敏感性已按 SPEC §27.1 R2 报过一次（报告一个数，不设门）。
- 怎么选：用 50 时必须带上「未解决/括号内」标签；报敏感性两端。
- 陷阱：**同一段 `[meta]` 里的命名冲突警告** —— 主文里唯一的「approximately 50」是 **PE 填料的 ε_r**（SI Table 2 的自由旋钮，见 §G.6），**不是** κ_b。若见到把 κ_b=50 引到那句话上的说法，那个引用是错的。**V2-02.1 让这条冲突变成了数字冲突**：同一个 B1 算例里，主体走 `kappa_for(50)` = 真 ε_b **500**，而 PE 填料走 SI 换算的真 ε_r **50** —— 一个算例内相差 10 倍（U2，`reports/archive/v1/B1_REPORT.md` 的 V2-02.1 caveat；是否重跑是用户裁决 D15）`[DIAL V2-02.1]`。另：κ_b **完全不进** 0D 拟合/闭合（均匀单畴态 D(k=0)=0，SPEC §1.3）。

**三阱 deck `pbzo_triplewell_yang2025` 的闭式阈值** ｜ E_F, E_A, P_max ｜ deck 电场（V/m ×1e-9）／ deck P
- 物理含义：单 P 六次势 `f(P) = a1P² + a11P⁴ + a111P⁶`（`loopfit.py:34`）的两个拐点（spinodal）给出双回线的正向/反向阈值场。
- 进入的公式：`loopfit.py:39` `E(P) = f'(P) = 2a1P + 4a11P³ + 6a111P⁵`；`:54` `spinodal_points`、`:68` `thresholds`、`:79` `p_max_at_field`。
- 典型范围与文献值（deck `[meta].e_thresholds_closed_form` 逐字）`[toml-meta]`：`P1s = sqrt(u1) = 0.187876095`、`P2s = sqrt(u2) = 0.359307737` deck；`E_F = E(P1s) = 6.687335938e-2` deck（66.873 MV/m）、`E_A = E(P2s) = 3.632441340e-2` deck（36.324 MV/m）；`E_F > E_A > 0`，双回线顺序正确；在 `E_max = 0.10` deck（100 MV/m）下 `P_max = 0.454636559` deck。Landau 系数：`0.2824, -1.698, 2.3145, 4.1314, -1.5456, -0.8443` `[toml-meta]`，源 Yang, Liu, Jiang et al., Nature **637**, 1104–1110 (2025), doi:10.1038/s41586-024-08505-7, SI `[pdf]`（`literature/yang2025_natur637_main.pdf` + `_SI.pdf`；`reports/LIT_yang2025_main.md` `[LIT]`）。
- 怎么选：这三个数是 G21b leg A 的 0D Newton 连续化必须复现到 ≤0.1% 的参考靶（SPEC §27.1 R4）。
- 陷阱：**0D 阈值 ≠ 空间阈值**。SPEC §27.1 R4(B) 已裁定：`E_A` 只**报告**、不设门 —— 空间成核会跑在 0D spinodal 之前（S1 的发现）。拿 0D 的 `E_A` 去卡空间跑的结果会假失败。

**`E_F` / `E_A` / `P_max` / `E_max`（`loopfit.fit` 的靶）** ｜ — ｜ deck 电场 ／ deck P
- 物理含义：反解的输入 —— 由**测得的**回线阈值、饱和极化与所用驱动场，求 `(a1, a11, a111)`。
- 进入的公式：`loopfit.py:153` `fit(E_F, E_A, P_max, E_max, chi_AFE=None, seeds=None, tol=1e-13)`；残差 `:131` `_residual_3`。
- 典型范围与文献值：`[LIT]`（测得的回线阈值）；本仓库在册的一组即上条的 Yang deck 数。
- 怎么选：SPEC §26.2 规定这四个是**强制**靶。拟合范围限于单轴族 `{a1, a11, a111}`，离轴 `a12/a112/a123` **冻结**到一个具名基准 deck（默认 Yang Table 1）并记进 `[meta]`（SPEC §27.1 R5）。
- 陷阱：`P_max` 是「在 `E_max` 下」的值 —— 两者必须成对给。

**`chi_AFE`** ｜ χ_AFE ｜ deck（1/(2a1) 的量纲）
- 物理含义：可选的第 4 个靶 —— P=0、E=0 处的小信号极化率 `χ = 1/f''(0) = 1/(2a1)`（`loopfit.py:44`）。
- 进入的公式：`loopfit.py:146` `_residual_4`。
- 典型范围与文献值：给了才用，`[LIT]`；默认 `None`。
- 怎么选：给它会把系统从 3 方程/3 未知变成 **4 方程/3 未知的最小二乘**（χ 残差一般非零）—— 想要精确命中三个阈值就别给。
- 陷阱：χ 只依赖 `a1`，所以它与 `E_F/E_A` 的信息高度重叠；加进去主要是当一致性检验，不是加自由度。

**`seeds` / `tol`（`loopfit.fit`）** ｜ — ｜ 无量纲 ／ 相对
- 物理含义：确定性根搜索的固定初值表与容差 —— **无 RNG**。
- 进入的公式：`loopfit.py:125` `_DEFAULT_SEEDS = [(0.3,-2.0,4.0), (0.2,-1.0,2.0), (0.1,-0.5,1.0), (0.5,-3.0,6.0), (0.3,-1.5,3.0), (1.0,-5.0,10.0)]`；`:169`。
- 典型范围与文献值：`tol = 1e-13`（numerical-only）；种子表 `[DIAL §26.0(a)]`。
- 怎么选：按设计确定性 —— 换种子表要连同 `[meta]` 一起记（SPEC §26.0(a)：method/targets/residuals 全部随返回 dict 进 deck `[meta]`）。
- 陷阱：名字叫 `seeds` 但**不是 RNG 种子**，是 `x0` 初值三元组；照 §G 的种子直觉去改会误解。

**全局极小约束（`check_global_min_constraint`）** ｜ — ｜ —
- 物理含义：AFE 的定义性约束 —— `f(sqrt(u2)) > 0 = f(0)`，即 `P=0` 必须仍是**全局**极小（否则那是铁电，不是反铁电）。
- 进入的公式：SPEC §27.1 R1/R5；`loopfit.py:91`。
- 典型范围与文献值：SPEC 声明的物理约束，非数值参数。
- 怎么选：不可关。违反的候选解被**丢弃**（不返回）；若所有种子的收敛点都违反，函数抛 `TripleWellConstraintError` 并先打印违规根与能量。
- 陷阱：它必须**大声失败**。G21a 里专门有 ≥1 组贴着约束边界的合成靶来跑这条拒绝路径；把异常吞掉等于让拟合器悄悄产出铁电 deck。

---

## §I 八面体倾转 theta 扇区（`solvers/octa_tilt.py`）

**物理前言（≤15 行）**。`theta(r)` 是 R 点氧八面体转动的三分量赝矢量序参量（分量 i = 绕 i 轴的转动），即反铁畸变（AFD）自由度。与 §H 的 `q` 同族的两条**载荷性**约定（SPEC §28.1，门测）：θ **不**产生泊松/退极化场；E **只**通过 `f_Ptheta` 经 P 间接耦合到 θ。自由能加项：`f_theta = b1|θ|² + b11 Σᵢθᵢ⁴ + b12 Σ_{i<j}θᵢ²θⱼ²`，`f_gradtheta = ½·G_θ|∇θ|²`，可选转伸缩 `ε*_ij += Λ^θ_ijkl θ_kθ_l`。
**θ 有两套 deck 单位约定，是 per-deck 而非全局的**（SPEC §28.5 R2 / §28.7 R2 均明写）：**模式 A（米）** —— θ 是**线位移**（Chen 在 Rabe App A 的原文），原始 SI 值 ~1e-10–1e-11 m 会比 P 低 9–10 个数量级，故按引擎既有的 nm 长度单位重标；**模式 B（度）** —— Haun 的 Ferroelectrics 99 把 θ 定义为**角度（度）**，本身已是 O(1–10)，保持不缩放（与 P 同为「unscaled field」）。

**`theta` 的单位与换算** ｜ θ ｜ 模式 A：deck 为 nm 数（SI = m，`si_to_deck_theta` ÷1e-9）；模式 B：度（SI == deck，`si_to_deck_theta_angle_deg` 为恒等）
- 物理含义：见前言。
- 进入的公式：`units.py:150` `THETA_LEN_SCALE = 1e-9`、`:155` `si_to_deck_theta`、`:172` `si_to_deck_theta_angle_deg`、`:196` `si_to_deck_theta_landau(value_si, theta_power)`（因子 `× (1e-9)^m × 1e-9`，`m` = 该项里 θ 的**总幂次**：`b1θ²` 取 2、`b11θ⁴` 取 4、`t11P²θ²` 取 **2**（P 不贡献额外因子））；转换点 `octa_tilt.py:157` `sto_afd_landau_deck`。
- 典型范围与文献值：模式 A 下真实 STO 倾转位移 ~1–10 pm → deck θ ~0.01–0.1（`units.py:155` docstring）；模式 B 下 Haun Table I/II 的角度 ~1–7° `[pdf]`（`literature/haun1989_ferro99_p4_tilt.pdf`、`_p5_theory.pdf`；`reports/LIT_haun_tilt.md` `[LIT]`）。
- 怎么选：**跟 deck 走**，不要跨 deck 搬系数。`THETA_LEN_SCALE = 1e-9` 是被标记的选择（`[DIAL §28.5 R2]`：复用既有 nm 约定，"flagged, not silent"）。
- 陷阱：三个坑。(1) `si_to_deck_theta_landau` **只用于米模式**；度模式直接用 `si_to_deck_landau`（`units.py:188-195` 注释明写）。(2) `theta_power` 是**总幂次**，耦合项 `t11P²θ²` 传 2 而不是 4 —— g22c 的有限差分门会抓错幂次。(3) 米模式与度模式的 `b11` 数值相差 10 个数量级以上（169000 vs 5.0791e-06），肉眼「不合理」的那个可能才是对的。

**`b1` / `b11` / `b12`（`b111` / `b112` / `b123`）** ｜ β₁, β₁₁, β₁₂ ｜ 能量密度，随 θ 约定而定（见上条）
- 物理含义：θ 自己的 Landau 势 —— `b1` 控转变（`b1 < 0` 时倾转失稳），`b11`/`b12` 是四阶各向异性，选出转动轴的取向。
- 进入的公式：SPEC §28.1；`octa_tilt.py:367` `_explicit_N_theta` 里的 `solver.bulk_force(theta, alpha_theta)`（θ 复用 P 的 Devonshire 机器），能量在 `:414` `total_energy_ptheta`。
- 典型范围与文献值：三套 deck 的 `b1/b11/b12`、`t11/t12/t44`、`Gtheta` 与 `Λ^θ`
  由 `tools/param_tables.py` 从 `params/*.toml` 生成 —— **见 §K 表 K.9**（含每套的 θ
  单位约定与源级）。
  **`b111`/`b112`/`b123` 在三个 deck 里全部为 0**（按约定；θ 势只到四阶）。
- 怎么选：`sto_afd_rabeA` 是 T=298 K 的**快照** —— `[meta]` 记录 α₁(T) 与 β₁(T) 在源里是 coth 型闭式，换温度要重算而不是插值。`pztG` 借的是这套**闭式**而不是这组数：census T3 记为「T_PZT=100 K，θ 在 50 K 取值」，所以它的 `b1 = −17.0001` 与 stoA 的 `142.56` 不同而 `b11/b12` 相同，不是转录错误。SPEC §28.5 R3 的候选散布政策：Table 3 里有 2–4 个未区分数值的行**默认取第一个**，全散布作为括号记进 `[meta]`，并给一行解析式（如 `θ(T) ∝ b11^{-1/2}`）。
- 陷阱：(1) `[meta]` 记录了**书里自己的单位排印错误**：`b11` 行印作 N·m⁻⁴，量纲分析与 `b12` 行都要求 N·m⁻⁶。(2) Rabe App A 的 STO Table 3 各行引用 refs [10]–[15] **未逐行区分**，因此**没有任何一个 STO 系数能归属到具体某篇论文**（T6 #20）。(3) `pzt_r3c_g22e_model` 是**模型材料复合体**，不是文献 PZT-倾转 deck。

**`t11` / `t12` / `t44`** ｜ t₁₁, t₁₂, t₄₄ ｜ 能量密度，随 θ 约定而定
- 物理含义：P–θ **双二次**耦合。实现的符号形式（`octa_tilt.py:228-234`，SPEC §28.5 R2 用 Chen 印出的**减号前缀**式**取代**了 §28.1 原来的「+」约定）：
  `f_Pθ = −t11·Σᵢ Pᵢ²θᵢ² − t12·Σ_{i≠j} Pᵢ²θⱼ² − t44·(P₁P₂θ₁θ₂ + P₂P₃θ₂θ₃ + P₃P₁θ₃θ₁)`。
  表中 t11/t12 为负 ⇒ 净正的四阶交叉 ⇒ **竞争性**（正确的 STO 物理）。
- 进入的公式：`octa_tilt.py:242`（能量）、`:248`（δf/δP）、`:254`（δf/δθ），键名表 `:236` `COUPLE_KEYS = ("t11","t12","t44")`。
- 典型范围与文献值 `[toml-meta]`：`stoA` 与 `pztG`：`−174 / −75.5 / 585`（米约定，Rabe App A Table 3）；`pztH`：`−4.30182e−3 / −1.86659e−3 / 1.4463e−2`（度约定，Haun Ferroelectrics 99，`reports/LIT_haun_tilt.md` `[LIT]`）。
- 怎么选：`t44`（`PᵢPⱼθᵢθⱼ` 族）是 SPEC §28.5 R2 为匹配源文献闭合而**扩进**菜单的槽位。符号括号：SPEC §28.5 R5 记录 Kornev et al., PRL **97**, 157601 (2006)（PZT 中为协同性，开放获取、非 Haun 谱系）`[std-ref 未复核]`；G22e 因此对借用耦合的**两个符号都算** 0D 解析、报告哪个给出 R3c 型共存。
- 陷阱：`P ≡ 0` 时 `couple_force_theta` 对**任意**耦合系数都恰为 0（`octa_tilt.py:367-369` docstring，SPEC G22d(ii)）—— 这是接线自检，不是物理结论；用它验证「θ 不被 E 直接驱动」这条约定。

**`Gtheta`** ｜ G_θ ｜ 同 G₁₁ ／ `si_to_deck_gradient` ×1e9
- 物理含义：θ 的各向同性梯度刚度，`f_gradθ = ½·G_θ|∇θ|²`（三分量求和）。
- 进入的公式：SPEC §28.1；`octa_tilt.py:265` `build_ops_theta`（`Aθ(k) = G_θ(kx²+ky²)`，标量、三分量相同、无 D(k)、无 G12/G44/G44p 交叉项）、`:271` 能量密度。
- 典型范围与文献值：三个 deck **全部 = 1.0** `[DIAL §28.5]`。**范围：仓库内无来源**（T6 #4，与 `Gq` 同因：没有任何来源发表过 θ 梯度系数）。
- 怎么选：只能声明为旋钮；与 `Gq` 完全同一条处置规则。
- 陷阱：同 `Gq` —— 借的是 BTO 那个本身就不是测量值的 `G110 = 1.0`（T6 #1），两级都要说明。

**`Lam11_theta` / `Lam12_theta` / `Lam44_theta`** ｜ Λ^θ ｜ m⁴/C² 式（SI == deck，无换算）
- 物理含义：**转伸缩**（rotostriction）本征应变 `ε*_ij += Λ^θ_ijkl θ_kθ_l`。
- 进入的公式：SPEC §28.1；`octa_tilt.py:297` `with_theta_eigenstrain`、`:304` `elastic_force_explicit_theta`（符号自检 `:316`）。
- 典型范围与文献值：**只有 `sto_afd_rabeA` 有**：`8.7 / −7.8 / −9.2` `[toml-meta]`（Rabe App A Table 3）。**其他材料范围：仓库内无来源**（T6 #14）。
- 怎么选：**按 SPEC §28.5 R6 停放（PARKED）** —— 弹性在 WP18 全程 OFF，Table 3 的 Λ 与 c_ij 行只是逐字停放供未来的应变耦合通道用。
- 陷阱：deck 里有值不等于路径被验证过。启用前先补该通道自己的 FD 一致性门。

**θ 通道的未决前提（必须随任何 R3c 结论一起写出）**
- `pzt_r3c_g22e_model` / `pzt_r3c_haun_x02` 的 (x, T) 是否真的落在 **R3c** 相内，**仓库内无来源**（T6 #24）：Corker, Glazer, Whatmore, Stallard & Fauth, JPCM **10** (1998), doi:10.1088/0953-8984/10/28/007 —— R3m/R3c 边界的那篇 —— **不在手**（`literature/README.md` 的 "Wanted" 第 4 优先）。SPEC §28.5 R5 的原话：锚点 (x,T) 必须在 R3c 内被论证，`x=0.6@298K` **不能**假定为 R3c。因此 §28.5 R5 把 G22e 的结论限定为**结构性/定性**（[111] 方向 P 与 θ 同时非零），**永远不是一个倾转角数值**。
- 两个 θ-通道 PZT deck **完全没有 `[elastic]` / `[electrostriction]` 段**（T6 #7）。

---

## §J 协议参数（`solvers/energy_protocol.py`、`loopfit.py`、`gates/g21b_*`、薄膜通道 `film3d.py`/`film_elastic.py`）

**物理前言（≤15 行）**。储能协议把一条**状态连续**的驱动路径 `E: 0 → E_max → 0` 施加到系统上，记录 (E, ⟨P⟩) 轨迹，再对轨迹做梯形积分：`W_store = ∫E dP`（充电支）、`W_rec = ∫E dP`（放电支，Yang 主文 p.1104 的 Ue 定义）、`W_loss = W_store − W_rec`（= 闭合叶面积）、`η = W_rec/(W_rec + W_loss)`。**协议本身是可引用性的前提**：SPEC §29.1 规定两条协议都要实现、每个输出都要贴上自己的协议标签 ——(i) 清洁弛豫（确定性），(ii) Yang 的噪声初态。这是 S1 的 G20e(i) 发现直接生出的规矩：**来自未声明协议的回线面积/W_rec 类数字不可引用**。速率是本扇区的核心自由参数，用 `n_dwell` 表达，靠**阶梯收敛**而不是靠物理速率来定标（T6 #15：仓库内没有任何有出处的实验场速率）。

**`N` / `delta_nm`（`build_case`）** ｜ N, Δ ｜ 格点数 ／ nm
- 物理含义：N×N 方盒与格距，`L = N·Δ`。
- 进入的公式：`energy_protocol.py:52` `build_case(N=64, delta_nm=1.0, kappa_b=None, params_path=None, *, eps_b=None, permittivity_deck=None)`（V2-02 E14 加的后两个是 keyword-only，见下一条 `kappa_b`）。
- 典型范围与文献值：默认 `64` / `1.0` `[DIAL §26.4(b)]`（调试期定尺）；SPEC §26.2 记录的 Yang 工作默认为 128²/Δ=1 nm，SPEC §27.1 R2 明确它是**工作默认**、「unverifiable against paywalled Methods within budget」。
- 怎么选：64² 默认 + 一次 128² 的网格不敏感性抽查（SPEC §27.1 R4）。
- 陷阱：`params_path=None` 会静默解析到 `params/pbzo_triplewell_yang2025.toml`，`kappa_b=None` 静默解析到 50（乘子；**真 ε_b = 500**，V2-02.1）—— 换材料时两个都要显式传，新算例直接用 `eps_b=`。

**`kappa_b` / `eps_b` / `permittivity_deck`（`build_case` / `kappa_for`）** ｜ κ_b, ε_b ｜ 乘子（deck 介电率 = `κ_b × 0.08854187`）／ 无量纲 ε_b ／ deck κ
- 见 **§H `kappa_b`** 的完整条目（T6 #5、命名冲突、括号 [25,100]）。此处只补一句进入点：`energy_protocol.py:113` `kappa_for`，基常数 `:49` `_KAPPA_UNIT_EPSB10 = 0.08854187`（V2-02.1 改名，旧名 `_EPS0_DECK_BASE`） —— 这个基常数**不是 ε₀**，它等于 10ε₀，所以 `kappa_for(m)` 给出的是 **ε_b = 10·m** `[DIAL V2-02.1]`。
- **V2-02 新增的两个关键字（E14，旧路径逐位不变）**：`build_case(eps_b=…)` 收**SI 的相对背景介电常数**，内部走 `units.relative_to_deck_permittivity`（κ = 0.0088541878128·ε_b）；`build_case(permittivity_deck=…)` 收**已经换算好的 deck κ**，原样用。三者互斥，只能传一个；不传就还是 `kappa_b` 的老默认（50，即真 ε_b=500）。返回的 case 字典多一个 `permittivity_convention` 键，值是 `"SI_relative"` / `"deck_absolute"` / `"legacy_0.08854187"` 三者之一，记录这次走的是哪条入口 —— 算例报 κ 时照抄它，不要再自己写「ε_b=50」这类标签。
- 怎么选：新算例一律用 `eps_b=`（意思唯一）；复现历史数字才用 `kappa_b=`。

**`n_equil` / `seed`（`ic_clean`）** ｜ — ｜ 步数 ／ 整数
- 物理含义：协议 (i) 清洁弛豫初态 —— 先无场弛豫 `n_equil` 步得到一个确定性的 t=0 态。
- 进入的公式：SPEC §29.1(i)；`energy_protocol.py:89` `ic_clean(grid, alpha, ops, dt, seed=0, n_equil=2000)`。
- 典型范围与文献值：`n_equil = 2000`、`seed = 0` `[DIAL §29.1(i)]`。**无文献来源**。
- 怎么选：确认弛豫已到平台（这条是 WP20 G21b 的先例协议）；改了要重报。
- 陷阱：与协议 (ii) **不可混用于同一张图**；SPEC §29.1 要求每个输出都带协议标签。

**`amplitude`（`YANG_NOISE_AMPLITUDE_SI`）/ `seed`（`ic_noise_yang`）** ｜ — ｜ C/m²（映到 deck）／ 整数
- 物理含义：协议 (ii) —— 一次性随机噪声**本身就是** t=0 态，直接从它开始扫场，**不**预弛豫（与 (i) 的关键差别，docstring 明写）。Yang 原话：「initial polarization nuclei consisted of randomized noise」。
- 进入的公式：SPEC §27.3(d)/§29.1(ii)；`energy_protocol.py:106` → `solver.random_smooth_field(grid, 3, amplitude=YANG_NOISE_AMPLITUDE_SI, kmax_frac=0.3, seed=seed)`；常量在 `:25`。
- 典型范围与文献值：`YANG_NOISE_AMPLITUDE_SI = 0.001` `[LIT]`（SPEC §29.1(ii) 记为 "amplitude 0.001 C/m^2 mapped to deck units"；`reports/LIT_yang2025_main.md`）。`kmax_frac = 0.3` 是代码默认 `[DIAL]`。
- 怎么选：SPEC §29.2 G23c 要求**每条协议 × 3 个噪声种子**并报散布。
- 陷阱：噪声初态使 `E_A`（放电/反向支）成为成核控制量 —— 两条协议在放电支上的差异正是 G23c 要回答的问题之一，不是数值噪声。

**`n1`（`e_path_unipolar`）** ｜ n₁ ｜ 每半程的场点数
- 物理含义：单极路径 `0 → e_max`（n₁ 点）`→ 0`（n₁ 点）的场分辨率。
- 进入的公式：`energy_protocol.py:114`、`:118` `np.concatenate([linspace(0,e_max,n1+1)[1:], linspace(e_max,0,n1+1)[1:]])`。
- 典型范围与文献值：无默认值；g21b 用 `N1 = 15`（`gates/g21b_loop_closure.py:55`，注为 "WP17 convention"）`[DIAL]`。**范围：仓库内无来源**（T6 #15）。
- 怎么选：够密到梯形积分不吃掉阈值跳变即可；与 `n_dwell` 一起做收敛。
- 陷阱：**E=0 的起点不含在 `e_path` 里** —— `run_unipolar` 自己把初态的 E=0 状态放在索引 0（与 g21b 的 `e_path_half` 完全一致）。自行拼路径时重复加 0 会让 `energy_integrals` 的分支簿记（长度必须是 `2n1+1`）错位。

**`e_max` / `E_max`** ｜ E_max ｜ deck 电场 = V/m ×1e-9
- 物理含义：单极驱动的峰值场。
- 进入的公式：`energy_protocol.py:114`；门里 `gates/g21b_loop_closure.py:54` `E_MAX = 0.10`。
- 典型范围与文献值：`0.10` deck = **100 MV/m** `[toml-meta]`（与 `pbzoT [meta]` 的闭式参考场同一个数）。SPEC §29.2 G23b 要求一次 `E_max` 在阈值**以下**（如 `0.8·E_F`）的可逆支检验。
- 怎么选：按目标物理选（阈下 = 可逆支；阈上 = 滞回支）；换 deck 时用该 deck 自己的 `E_F` 标定。
- 陷阱：deck 电场因子 ×1e-9 是**导出量**，不是纯约定（V2-02.1；函数 `units.si_to_deck_field`，门 K5 `permittivity_convention` Leg C，T6 #12 已关闭，见 §A.4 坑 3）；`0.10 deck ↔ 1e8 V/m` 这条交叉核对仍然成立。

**`n_dwell`（`run_unipolar`）** ｜ — ｜ 每个场点保持的步数
- 物理含义：**速率旋钮** —— 每个 E 值上保持 `n_dwell` 步，越大越准静态。这是本仓库表达「扫场速率」的唯一方式。
- 进入的公式：SPEC §29；`energy_protocol.py:121` `run_unipolar(alpha, ops, Minv, dt, n_dwell, P_init, e_path, record_fields=False)`。
- 典型范围与文献值：无默认值。g21b 的速率阶梯 `RATE_LADDER = (3000, 6000, 12000)`、`RATE_FINAL = 12000`、`RATE_128_CHECK = 6000`（`gates/g21b_loop_closure.py:59-61`）`[DIAL §26.4(d)]`。**范围：仓库内无来源** —— T6 #15：没有任何有出处的实验场速率，回线速率是协议的自由参数。
- 怎么选：SPEC §26.4(d) 的**最粗判决性阶梯**（2–3 点，不是生产阶梯）；SPEC §29.2 G23c 的判据是**最高两级之间 ≤2%**。SPEC §29.2 的 RULING RIDER：只有在某条**已声明**协议下单极阶梯收敛 ≤2%，G20e(i) 对该 deck 的 W_rec 隔离才解除，`W_rec/η` 才变成可引用（且必须带协议标签）；否则隔离继续。
- 陷阱：`n_dwell` 与 `dt` 一起决定物理时间 —— 换 `dt` 而不换 `n_dwell` 就等于**悄悄换了速率**，阶梯收敛结论随之作废。

**`energy_integrals(E_full, P1_full, n1)`（W_store / W_rec / W_loss / η）** ｜ — ｜ deck 能量密度
- 物理含义：梯形积分 + 显式分支簿记（G23a 门测）。`E_full/P1_full` = `[E=0] + 充电(n1 点) + 放电(n1 点)`，长度 `2n1+1`。
- 进入的公式：SPEC §29.1；`energy_protocol.py:147`，`:160` `W_store = trapezoid(E_chg, P_chg)`、`:161` `W_rec = -trapezoid(E_dis, P_dis)`、`:163` `η`。
- 典型范围与文献值：G23a 的解析锚 —— 纯二次 deck（`a1>0`，高阶全 0）有闭式 `P = χE`、`W_store = W_rec = ½χE_max²`、`W_loss = 0`、`η = 1`；引擎+积分器需复现 `W_rec ≤0.1%`、`W_loss ≤1e-4·W_rec`。
- 怎么选：这是仪器的绝对锚，任何改动先跑它。
- 陷阱：**积的是 P，不是 D**。轨迹记录的是 `P1_full = ⟨P₁⟩`（`energy_protocol.py:130`），积分 `∫E dP` 里**没有** `κ_b·E` 的背景项 —— 与实验上按 D=ε₀ε_rE+P 定义的储能密度**不是同一个量**，跨文献比较时必须说明。另一个坑：`W_rec` 的**积分限被反过来**（Yang 的 Ue 是 `∫[Pr→Pm] E dP`，而放电轨迹自己走的是 Pm→Pr），所以代码里显式取了负号；照抄 `trapezoid` 而不取负会得到负的「可回收能量」。

**`verified_dt` 的参数（`T_total` / `tol` / `dt0` / `max_halvings` / `n1_probe` / `n_dwell_probe`）** ｜ — ｜ deck 时间 ／ 相对 ／ deck 时间 ／ 计数
- 物理含义：**固定总物理时间**的 dt-vs-dt/2 收敛验证。协议版签名 `energy_protocol.py:167` `verified_dt(alpha, ops, grid, e_max, dt0, n1_probe=8, n_dwell_probe=200, T_total=None, tol=0.01, max_halvings=10, P_init_fn=None)`；通用版 `timestep.py:171`（`dt0=0.02, n_probe=200, growth=2.0, p_cap=None, max_doublings=30, scheme='sbdf1', tol=0.01, max_halvings=10, eps=1e-12`）。
- 进入的公式：SPEC §28.6(a)/§29.3；三法见 SPEC §36.3(2C)。
- 典型范围与文献值：`tol = 0.01` `[DIAL §28.6(a)]`；`dt0 = 0.02`、`n_probe = 200`、`growth = 2.0` `[DIAL §2]`；`n1_probe = 8`、`n_dwell_probe = 200` `[DIAL §29.3]`；`max_halvings = 10`、`eps = 1e-12`（numerical-only）。
- 怎么选：**每个 deck × 协议组合**跑一次再上生产（SPEC §29.3）。
- 陷阱：`verified_dt` 的 docstring 记录了一次真实事故 —— **固定步数**的减半循环给出**假收敛信号**（更小的 dt 积了更少的总物理时间，回落向未弛豫的初态而不是真轨迹，WP18 发现）。所以 `T_total` 必须给；这也是 §36.3(2C) 三法之一「fixed-total-physical-time」的由来。第三法「production forcing in the probe」由 §32.5 补入 —— 探针必须带上生产用的驱动。

**双极扫场与速率阶梯（g21b 式）** ｜ — ｜ —
- 物理含义：`e_path_full`（`gates/g21b_loop_closure.py:174`）走 `0 → +E_max → −E_max → 0`，一次给出四个分支；`e_path_half`（`:170`）走 `0 → +E_max → 0`，两个阈值都落在同一极性上。
- 进入的公式：与 `run_unipolar`/`run_sweep`（`:182`）同一条驱动通道。
- 典型范围与文献值：SPEC §27.1 R4(B) 的裁定：**最粗阶梯（2–3 个速率）跑半回线**，只在最终速率跑**一次**完整双极作为交付图；`E_MAX = 0.10`、`N1 = 15`、`RATE_LADDER = (3000,6000,12000)` `[DIAL §26.4(d)]`。
- 怎么选：半回线做阶梯（便宜且两个阈值都在），全回线只出图。`RATE_FINAL` 须**先验固定**为阶梯的最大速率（`:60`）—— 不能事后挑一个好看的。
- 陷阱：SPEC §27.1 R4(B) 明确 `E_F` 与 `P_max` 对 0D 设门 ≤2%，而 `E_A` **只报告不设门**（空间成核跑在 0D spinodal 之前）；回线面积属**隔离类**，未声明种子协议的不可作为 W_rec 引用。

**`seeds` / `tol`（`loopfit.fit`）** —— 见 §H 同名条目（确定性初值表、无 RNG、`tol = 1e-13`）。

### J.1 薄膜通道的协议常量（SPEC §13）

**`paper_alpha(T_C)`** ｜ T ｜ **摄氏度（°C）**
- 物理含义：薄膜通道在给定温度下的 α 字典。
- 进入的公式：`film3d.py:233`（模式同 `cases/film_tier2.py` 的 `alpha_at()`）。
- 典型范围与文献值：deck `bto_LCC2005_paper` 的 `[meta]` `[toml-meta]`；源 Li & Chen, APL **88**, 072905 (2006), Ref. 18 脚注（力学）+ Eq. 3（Landau）+ Eq. 7（各向同性梯度）`[std-ref 未复核]`。
- 怎么选：按算例温度。
- 陷阱：**这是全仓库唯一以摄氏度为入口的函数**（census T1b 明确标注 "NOT K"）。传 298 会得到 298 °C。

**`paper_grid(Nx, Ny, Nz, dx_nm)` / `PAPER_DX_NM` / `PAPER_HF_DX` / `PAPER_NZ`** ｜ — ｜ 计数 ／ nm
- 物理含义：论文自己的薄膜网格 —— 面内 128×128、z 向 21 个节点跨越膜厚 `hf = 20·dx`。
- 进入的公式：`film3d.py:215` `paper_grid(Nx=128, Ny=128, Nz=21, dx_nm=0.4)`；常量 `:189-191`。
- 典型范围与文献值：`PAPER_DX_NM = 0.4`（`[DIAL §13.0]`，注为「WP6 Tier 2's choice, reused verbatim」）；`PAPER_HF_DX = 20`、`PAPER_NZ = 21` —— 论文正文的 "hf = 20·dx" `[LIT]`（SPEC §13.0 引用了原文）。
- 怎么选：复现论文时整套用，别单换一个。
- 陷阱：(1) `film3d.make_grid_3d(Nx, Ny, Lx, Ly, Nz, hf)` 与 `bulk3d.make_grid_3d(Nx, Ny, Nz, Lx, Ly, Lz)` **同名不同序**（第 3 个位置参数一个是 `Lx`、一个是 `Nz`），positional 调用错模块会静默造出错误的盒子（`film3d.py:194` docstring 的 NAME COLLISION 警告，全表在 ENGINE_GUIDE §2.1）。(2) `bto_LCC2005_paper` 的 `G110 = 0.00593856` 是**由 dx=0.4 nm 反推**的 —— `[meta]` 记为「a documented free parameter, NOT tuned to any result」；改 dx 就等于改梯度系数。

**`KAPPA_B_DECK` vs `KAPPA_B_PAPER_LINEAGE`** ｜ κ_b ｜ deck 介电率
- 物理含义：薄膜通道背景介电常数的**两条谱系**：`KAPPA_B_DECK`（`film3d.py:272`）= `paper_mechanics()['permittivity']` = **0.0885419**，这**不是**「HM2006 继承的 ε_b=1 占位值」，而是 **Ferret 原值 ε_b=10**（Ferret 甲板原注释 "e_b = 10"）；`KAPPA_B_PAPER_LINEAGE`（`:273`）= `66.0 × KAPPA_B_DECK` = 5.84376…，即 **ε_b=660** —— 66× 乘在一个已经等于 10ε₀ 的数上，并不是 ref 25 / Haun 1987 的 ε_b=66（V2-02.1）`[DIAL V2-02.1]`。
- 进入的公式：`film3d.poisson_factor` 及 SPEC §13 的 κ_b 谱系（旧标 1 / 10 / 50 / 66 = 真 ε_b **10 / 100 / 500 / 660**）。
- 典型范围与文献值（下面是**文献自己给的 ε_b**，甲板实跑值见括号）：甲板原值 ε_b=10 `[toml-meta]`（Ferret 原注释；params 各甲板 `[meta].eps_b_true`）；文献 ε_b=66 `[std-ref 未复核]`（Haun 1987 / ref 25）→ 本仓库 `KAPPA_B_PAPER_LINEAGE` 实跑 660；文献 ε_b=10 `[std-ref 未复核]`（Wang, Ma, Li, Britson & Chen, Acta Mater. **61** (2013) 7591, p.7597，PZT/PMN/nnbm 用）→ 实跑 100；乘子 50 `[DIAL]`（见 §H `kappa_b`，T6 #5）→ 实跑 500 `[DIAL V2-02.1]`。
- 怎么选：**WP7b 裁定**（`bto_LCC2005_paper [meta].kappa_b_ruling`）：遗留门 G0–G10 **保留**甲板原值（真 ε_b=10）；论文复现算例用 66×（真 ε_b=660）。`film3d.py:264-271` 注释记录了理由 —— ref 31 本身读不到，所以 `KAPPA_B_DECK` 是每个门的默认，`KAPPA_B_PAPER_LINEAGE` 作为**有出处的敏感性对照报告出来，不静默替换**。
- 陷阱：这四个乘子里只有 10 和 66 在**文献侧**有出处，而**甲板侧的实跑值全部是它们的 10 倍**（V2-02.1）：任何 κ_b 结论都要同时标明走的是哪一条谱系、以及换算成 SI 的真 ε_b 是多少。论文复现算例按真 ε_b=66 重跑与否是用户裁决 **D15**，尚未执行。

**`hs_dx`** ｜ h_s ｜ dx 单位（整数节点数）
- 物理含义：弹性叠层里衬底的厚度（节点数）；`make_stack_grid` 把 film3d 的仅膜 z 网格向下延伸 `hs_dx` 个节点，节点 0（z=−hs）为 Dirichlet 并从求解中剔除，膜内节点编号不变。
- 进入的公式：SPEC §13.0（"hs=12dx"）；`film_elastic.py:98` `make_stack_grid(film_grid, hs_dx=PAPER_HS_DX)`，常量 `:89` `PAPER_HS_DX = 12`。
- 典型范围与文献值：`12` `[LIT]`（SPEC §13.0 引用论文原文）。
- 怎么选：按论文；改动要连同弹性收敛一起复验。
- 陷阱：单位是 **dx 的倍数**，不是 nm —— 换 `dx_nm` 会同时改变物理衬底厚度。

**`e0`** ｜ e₀ ｜ 无量纲失配应变
- 物理含义：Li 2002 的衬底夹持均匀应变形式 `ε̄ = diag(e0, e0, −2(C₁₂/C₁₁)e0)` —— 面内双轴失配加上泊松式的面外弛豫。
- 进入的公式：SPEC WP6 §12.1/§13.2；`film_elastic.py:301` `setup_film_elastic(..., e0)`、`:128` `homogeneous_strain_film_e0`；`elastic.py:222` `homogeneous_strain_film_full`。
- 典型范围与文献值：无默认值（必填）；形式出处 Li, Hu, Liu & Chen, Acta Mater. **50** (2002) 395 `[std-ref 未复核]`（SPEC §13 第 536 行引用）。具体 e₀ 值由算例的衬底/膜错配决定 —— WP19 的失配序列自行声明。
- 怎么选：由目标衬底的晶格错配算，不拟合。
- 陷阱：`e0` 与体材通道的 `eps_m`（`elastic.py:181/:210`，`ε̄₁₁=ε̄₂₂=ε_m`，SPEC §1.4(b)）是**两个不同的均匀应变模式**，还有第三个 `k0='stress_free'`（SPEC §1.4(a)）。三者不可混用；`k0` 选错会让整条弹性能量走另一条分支而不报错。

---

## §K 材料值总表(由 `tools/param_tables.py` 生成)

<!-- param_tables:begin -->

<!-- GENERATED by tools/param_tables.py from params/*.toml -- do not hand-edit
     inside these markers. Regenerate:  python3 tools/param_tables.py -->

本节的每一个数值由 `tools/param_tables.py` 从 10 个在用甲板 `params/*.toml` 与 9 个已存档甲板 `params/archive/*.toml` 读出(SPEC §42:“Material value tables are GENERATED … so they cannot drift”),物理解释在 §A–§J 的词条里。**改数请改甲板,然后重跑脚本。**

**标「(存档)」的甲板已存档**:没有任何计算读取它们,2026-11 移入 `params/archive/`(逐套原因与恢复方法见 `params/archive/README.md`)。它们的数值与出处照常列在下面各表(排在在用甲板之后),不删。

数字为甲板值的 **6 位有效数字**(`%.6g`,与普查 T3 同一精度);完整位数以 `params/*.toml` 为准。“源级”列是 Phase B 写作者按 SPEC §42 EVIDENCE LAW 标注的证据分级,随表一起由脚本内的映射带出(填自普查 T3),不随本表重排而丢失。

### K.1 逐甲板出处与文献支撑

“出处”是该甲板 `[meta]` 自己的引用,压到一行(首作者 / 期刊 / 年);“文献支撑”按普查 T3 的规则从 `reports/LIT_*.md` 与 `literature/README.md` 派生 —— PDF 是 gitignore 的本地文件,这里核对的是它在 `literature/README.md` 里的索引条目。两者都在生成时校验存在性,缺一即本脚本报错。

| 甲板 | 文件 | 出处([meta],短) | 文献支撑 |
|---|---|---|---|
| `bfoP` | `params/bfo_pan2019.toml` | Zhang 等, J. Appl. Phys. 101, 114105 (2007)(经 Pan 2019 SM p.3-4) | LIT: `pan2019_science`、`bfo_sources` · PDF: `pan2019_science_SM.pdf`、`zhang2007_jap101_bfo_strain.pdf`、`karpinsky2017_npjcm3_bfo_potential.pdf` |
| `bto118` | `params/bto_118K.toml` | Ferret 甲板 `runs/rhombo_walls/inputs/k_r109_L64.i`(非文献) | **无** LIT 抽取、`literature/` 无 PDF |
| `bto298` | `params/bto_298K.toml` | Ferret 甲板 `runs/rhombo_walls/inputs/k_t180.i`(非文献) | **无** LIT 抽取、`literature/` 无 PDF |
| `btoLp` | `params/bto_LCC2005_paper.toml` | Li & Chen, Appl. Phys. Lett. 88, 072905 (2006)(Ref.18 脚注 + Eq.3 + Eq.7) | **无** LIT 抽取、`literature/` 无 PDF |
| `btoP` | `params/bto_pan2019.toml` | Pan 等, Science 365, 578 (2019), SM p.3-4 | LIT: `pan2019_science`、`pan2019_forensics` · PDF: `pan2019_science_main.pdf`、`pan2019_science_SM.pdf` |
| `nnbm` | `params/nnbm_wang2026.toml` | Wang 等, Nano Energy 156, 112129 (2026) SI 表;网格/BC 取 Nat. Commun. 16, 2892 (2025) | LIT: `wang2026_nanbo3`、`wang2025_natcomm` · PDF: `wang2026_nanoen_nanbo3_main.pdf`、`wang2025_natcomm_main.pdf`、`wang2025_natcomm_SI.pdf` |
| `pbzoK` | `params/pbzo_kittel_lum2022.toml` | Lum, Lim & Chew, J. Phys.: Condens. Matter 34, 415702 (2022) Table 1(开放获取) | **无** LIT 抽取、`literature/` 无 PDF |
| `pbzoT` | `params/pbzo_triplewell_yang2025.toml` | Yang, Liu, Jiang 等, Nature 637, 1104–1110 (2025) SI | LIT: `yang2025_main`、`yang2025_figs` · PDF: `yang2025_natur637_main.pdf`、`yang2025_natur637_SI.pdf` |
| `stoA` | `params/sto_afd_rabeA.toml` | Rabe/Ahn/Triscone 2007 App A (Chen) Table 3(书 p.367) | **无** LIT 抽取、`literature/` 无 PDF |
| `stoP` | `params/sto_pan2019.toml` | Pan 等, Science 365, 578 (2019), SM p.3-4 | LIT: `pan2019_science`、`pan2019_forensics` · PDF: `pan2019_science_main.pdf`、`pan2019_science_SM.pdf` |
| `btoBC`(存档) | `params/archive/bto_BC1984_6th.toml` | Rabe/Ahn/Triscone 2007 App A (Chen) Table 1 “Sixth order”(书 p.365) | LIT: `pan2019_forensics` |
| `btoL8`(存档) | `params/archive/bto_LCC2005_8th.toml` | Rabe/Ahn/Triscone 2007 App A (Chen) Table 1 “Eight-order”(书 p.365) | **无** LIT 抽取、`literature/` 无 PDF |
| `pmn0`(存档) | `params/archive/pmn_clean_x0.toml` | Viehland, Jang, Cross & Wuttig, Phys. Rev. B 46, 8003 (1992) | **无** LIT 抽取、`literature/` 无 PDF |
| `pmnD`(存档) | `params/archive/pmn_disorder_R3.toml` | `generated_by = pmn_field.py` —— 不是材料甲板,是无序层的输入/输出记录 | **无** LIT 抽取、`literature/` 无 PDF |
| `pmnPT`(存档) | `params/archive/pmn_pt_landau_x0.30.toml` | Hangbo Zhang 等, Phys. Rev. B 96, 054109 (2017) —— 引用已确认,数从未转录(付费墙) | **无** LIT 抽取、`literature/` 无 PDF |
| `pto`(存档) | `params/archive/pto_Haun1987.toml` | Rabe/Ahn/Triscone 2007 App A (Chen) Table 5(书 p.368) | **无** LIT 抽取、`literature/` 无 PDF |
| `pzt06`(存档) | `params/archive/pzt_Haun_x0.6.toml` | Rabe/Ahn/Triscone 2007 App A (Chen) p.367 式组 + Table 4 柔度 | **无** LIT 抽取、`literature/` 无 PDF |
| `pztG`(存档) | `params/archive/pzt_r3c_g22e_model.toml` | 模型复合甲板(SPEC §28.5 R5):P 扇区 Rabe App A;θ 扇区借自 sto_afd_rabeA @ 50 K | **无** LIT 抽取、`literature/` 无 PDF |
| `pztH`(存档) | `params/archive/pzt_r3c_haun_x02.toml` | P 扇区 Rabe App A;θ 扇区 Haun, Furman, Halemane & Cross, Ferroelectrics 99, 55–62 (1989) | LIT: `haun_tilt` · PDF: `haun1989_ferro99_p4_tilt.pdf`、`haun1989_ferro99_p5_theory.pdf` |

### K.2 Landau 系数(6 阶)

顺序 `a1, a11, a12, a111, a112, a123`(TOML 键 `alpha1…alpha123`,引擎键映射见 SPEC §36.2 R2)。空白 = 该甲板没有这一项(**不是 0**)。

| 甲板 | T | a1 | a11 | a12 | a111 | a112 | a123 | 源级 |
|---|---|---|---|---|---|---|---|---|
| `bfoP` | 298 K | −0.39445 | 0.65 | 0.1 | 0 | 0 | 0 | `[pdf]` Zhang 2007(Pan SM 转录);α₁ 为 T=298 K 快照 |
| `bto118` | — | −0.0878442 | −1.49175 | 0.323 | 17.94 | 4.47 | 4.91 | `[toml-meta]` Ferret 甲板提取,**非文献**;`[meta]` 无 `T` 字段 —— 118 K 出自甲板 `k_r109_L64.i` 的世系,不是文件里的数(普查 T6 #21) |
| `bto298` | — | −0.027721 | −0.64755 | 0.323 | 8.004 | 4.47 | 4.91 | `[toml-meta]` Ferret 甲板提取,**非文献**;仓库基准甲板(G6a 往返靶);`[meta]` 无 `T` 字段 —— 298 K 出自甲板 `k_t180.i` 的世系(普查 T6 #21) |
| `btoLp` | 298 K | −0.037116 | −0.2097 | 0.7974 | 1.294 | −1.95 | −2.5009 | `[std-ref 未复核]` Li & Chen APL 88 Eq.3(== btoL8 那一套) |
| `btoP` | 298 K | −0.027722 | −0.64755 | 0.323 | 5.52 | 4.47 | 4.91 | `[pdf]`+`[LIT]` Pan 2019 SM;**α₁₁₁ 冲突未裁决**(T6 #18) |
| `nnbm` | 180 K(`landau_T_choice` 旋钮) | −0.005 | 0.56 | 0.289 | 1.65 | −0.866 | 31.9 | `[pdf]`+`[LIT]` Wang 2026 SI 表(T=180 K 为 `[DIAL §32]`) |
| `pbzoK` | 298 K | 0.1025 | −0.25 | 0 | 2.66667 | 0 | 0 | `[toml-meta]` Lum 2022 Table 1 的**精确 Kittel 变换**产物(`literature/` 无 PDF、无 LIT 抽取) |
| `pbzoT` | —([meta] 不给) | 0.2824 | −1.698 | 2.3145 | 4.1314 | −1.5456 | −0.8443 | `[pdf]`+`[LIT]` Yang 2025 SI(单 P 三阱工程甲板) |
| `stoA` | 298 K | 0.183166 | 1.7 | 1.37 | 0 | 0 | 0 | `[std-ref 未复核]` Rabe App A Table 3;refs [10]–[15] **逐行不可分辨**(T6 #20) |
| `stoP` | 298 K | 0.184619 | 1.7 | 1.37 | 0 | 0 | 0 | `[pdf]`+`[LIT]` Pan 2019 SM;与 stoA 的 α₁ 有 ±1.0 K 转录冲突(T6 #19) |
| `btoBC`(存档) | 298 K | −0.027722 | −0.64755 | 0.323 | 8.004 | 4.47 | 4.919 | `[std-ref 未复核]` Rabe App A Table 1 “Sixth order”(Bell & Cross 世系) |
| `btoL8`(存档) | 298 K | −0.037116 | −0.2097 | 0.7974 | 1.294 | −1.95 | −2.5009 | `[std-ref 未复核]` Rabe App A Table 1 “Eight-order”;`or_transition_note`:不适合 O–R 转变 |
| `pmn0`(存档) | 298 K | −0.0470587 | — | — | — | — | — | `[std-ref 未复核]` Viehland 1992 的 PMN 自身高温 Curie–Weiss 拟合;**α₁₁ 及以上故意缺失,不是 0**(T6 #9) |
| `pto`(存档) | 298 K | −0.17252 | −0.073 | 0.75 | 0.26 | 0.61 | −3.7 | `[std-ref 未复核]` Rabe App A Table 5 |
| `pzt06`(存档) | 298 K | −0.0834718 | 0.0361406 | 0.323308 | 0.185915 | 0.850273 | −4.06328 | `[std-ref 未复核]` Rabe App A p.367 式组,x=0.6 @ 298 K |
| `pztG`(存档) | 100 K | −0.128464 | 0.312911 | −0.00345387 | 0.428816 | 1.81422 | −7.54472 | `[std-ref 未复核]` P 扇区 Rabe App A(x_Ti=0.2, T=100 K) |
| `pztH`(存档) | 298 K | −0.0747709 | 0.312911 | −0.00345387 | 0.428816 | 1.81422 | −7.54472 | `[std-ref 未复核]` P 扇区 Rabe App A(x_Ti=0.2, T=298 K) |

**[landau] 段存在但为空的甲板:** `pmnPT`(存档) —— 每个字段都是 REQUIRED 占位,不是 0(普查 T6 #8)。

### K.3 Landau 八阶项(只列非零者)

| 甲板 | a1111 | a1112 | a1122 | a1123 | 源级 |
|---|---|---|---|---|---|
| `btoLp` | 38.63 | 25.29 | 16.37 | 13.67 | `[std-ref 未复核]` Li & Chen APL 88 Eq.3(== btoL8 那一套) |
| `btoL8`(存档) | 38.63 | 25.29 | 16.37 | 13.67 | `[std-ref 未复核]` Rabe App A Table 1 “Eight-order”;`or_transition_note`:不适合 O–R 转变 |

八阶键**默认 0.0**(`solver.ALPHA_KEYS_8TH`,solver.py:102):带键但全为 0 的甲板 —— `bto118`、`bto298`、`pbzoK`、`pbzoT`、`stoA`、`btoBC`(存档)、`pto`(存档)、`pzt06`(存档)、`pztG`(存档)、`pztH`(存档);
**根本没有这四个键**(genuinely ABSENT,SPEC §36.2 R1)的甲板 —— `bfoP`、`btoP`、`nnbm`、`stoP`、`pmn0`(存档)。

### K.4 梯度扇区(参考标度、比值、以及引擎实际用的绝对值)

绝对值 `G11/G12/G44/G44p` **不存在于任何甲板**:甲板给 `G110` 加四个无量纲比值,`fftpf/mat_registry.py:117` `_flat_to_deck` 在读取时相乘(mat_registry.py:140),且**只有** `G110` 与四个比值齐备时才产生 `deck["gradient"]`。本表按同一规则复算(ratio × G110),未 import `fftpf.mat_registry`(保持纯 stdlib)。

| 甲板 | G110 | G11/G110 | G12/G110 | G44/G110 | G44′/G110 | G11 | G12 | G44 | G44p | 源级 |
|---|---|---|---|---|---|---|---|---|---|---|
| `bfoP` | 0.0704 | 1.2 | 0 | 0.6 | 0.6 | 0.08448 | 0 | 0.04224 | 0.04224 | `[pdf]` Pan 2019 SM —— 仓库唯一有出处的绝对梯度 |
| `bto118` | 1 | 0.51 | −0.02 | 0.02 | 0 | 0.51 | −0.02 | 0.02 | 0 | `[DIAL]` G110=1.0 为重整化参考值 + Ferret 比值 |
| `bto298` | 1 | 0.51 | −0.02 | 0.02 | 0 | 0.51 | −0.02 | 0.02 | 0 | `[DIAL]` 同 bto118 |
| `btoLp` | 0.00593856 | 1 | 0 | 0.5 | 0.5 | 0.00593856 | 0 | 0.00296928 | 0.00296928 | `[toml-meta]` G110 由 dx=0.4 nm **反推**(“documented free parameter, NOT tuned”);比值为唯一的各向同性组 |
| `btoP` | 0.0704 | 1.2 | 0 | 0.6 | 0.6 | 0.08448 | 0 | 0.04224 | 0.04224 | `[pdf]` Pan 2019 SM(三个端元共享) |
| `nnbm` | −0.664019 | 1 | 0 | 0.5 | 0.5 | −0.664019 | 0 | −0.332009 | −0.332009 | `[DIAL §32]` G110(**负**)与 H11 联合标定;两篇源论文都不发表梯度数值 |
| `pbzoK` | 1 | 0.51 | −0.02 | 0.02 | 0 | 0.51 | −0.02 | 0.02 | 0 | `[DIAL]` 比值借自 BTO,`[meta].note` 标 not-validated |
| `pbzoT` | 1 | 0.51 | −0.02 | 0.02 | 0 | 0.51 | −0.02 | 0.02 | 0 | `[DIAL]` 比值借自 BTO |
| `stoA` | 1 | 0.51 | −0.02 | 0.02 | 0 | 0.51 | −0.02 | 0.02 | 0 | `[DIAL]` 比值借自 BTO |
| `stoP` | 0.0704 | 1.2 | 0 | 0.6 | 0.6 | 0.08448 | 0 | 0.04224 | 0.04224 | `[pdf]` Pan 2019 SM(三个端元共享) |
| `btoBC`(存档) | 1 | 0.51 | −0.02 | 0.02 | 0 | 0.51 | −0.02 | 0.02 | 0 | `[DIAL]` 力学/梯度/κ 借自 bto298(SPEC §9.1(a)) |
| `btoL8`(存档) | 1 | 0.51 | −0.02 | 0.02 | 0 | 0.51 | −0.02 | 0.02 | 0 | `[DIAL]` 借自 bto298 |
| `pto`(存档) | — | 0.51 | −0.02 | 0.02 | 0 | **无** | **无** | **无** | **无** | **无 G110**(占位必填);比值借自 BTO,非 PTO 验证 |
| `pzt06`(存档) | — | 0.51 | −0.02 | 0.02 | 0 | **无** | **无** | **无** | **无** | **无 G110**(占位必填);比值借自 BTO |
| `pztG`(存档) | 1 | 0.51 | −0.02 | 0.02 | 0 | 0.51 | −0.02 | 0.02 | 0 | `[DIAL]` 比值借自 BTO |
| `pztH`(存档) | 1 | 0.51 | −0.02 | 0.02 | 0 | 0.51 | −0.02 | 0.02 | 0 | `[DIAL]` 比值借自 BTO |

`G11`…`G44p` 列写 **无** 的甲板缺 `G110`,因此**没有绝对梯度扇区** —— 算例必须自己声明一个值并标为占位假设(SPEC §9.1(a))。

### K.5 背景介电率 `permittivity` → 引擎键 `kappa_b`

**真 ε_b = deck κ ÷ 0.0088541878128**(= ε₀×1e9,引擎单位制下真正的 deck ε₀;推导见 `reports/REVIEW_v2_02_units.md`,裁定 SPEC V2-02.1)`[DIAL V2-02.1]`。**旧标签倍数**列是 Part-I 一直印的 κ/0.08854187,它比真值整整小 **10 倍** —— 0.08854187 本身就是 ε_b=10(Ferret 原值,`BTO_wall_T298K.i:479-488`)。**deck κ 一个数没动(D7)**;旧世系 1/10/50/66 → 真值 10/100/500/660,见 §D.1。

| 甲板 | deck κ | **真 ε_b** = κ/0.0088541878128 | 旧标签倍数 κ/0.08854187 | 源级 |
|---|---|---|---|---|
| `bfoP` | 0.885419 | **100** | 10 | `[std-ref 未复核]` Wang 2013 Acta Mater 61 世系(文献 ε_b=10;本甲板 deck κ 实为 ε_b=100,V2-02.1) |
| `bto118` | 0.0885419 | **10** | 1 | `[DIAL §13]` Ferret/HM2006 原值(旧标 ε_b=1,实为 **10**,V2-02.1) |
| `bto298` | 0.0885419 | **10** | 1 | `[DIAL §13]` Ferret 原值(旧标 ε_b=1,实为 **10**,V2-02.1) |
| `btoLp` | 0.0885419 | **10** | 1 | `[DIAL §13]` Ferret 原值(旧标 ε_b=1,实为 **10**);`kappa_b_ruling` 的 66× 实为 ε_b=660(V2-02.1) |
| `btoP` | 0.885419 | **100** | 10 | `[std-ref 未复核]` Wang 2013 世系(文献 ε_b=10;deck 实为 100,V2-02.1) |
| `nnbm` | 0.885419 | **100** | 10 | `[std-ref 未复核]` Wang 2013 世系(文献 ε_b=10;deck 实为 100,V2-02.1) |
| `pbzoK` | 0.0885419 | **10** | 1 | `[DIAL §13]` Ferret 原值(旧标 ε_b=1,实为 **10**,V2-02.1) |
| `pbzoT` | 4.42709 | **500** | 50 | `[DIAL §26.2]` 乘子 50(deck 实为 ε_b=**500**,V2-02.1)—— **括号内旋钮**,主文只符号性定义 K(T6 #5) |
| `stoA` | 0.0885419 | **10** | 1 | `[DIAL §13]` Ferret 原值(旧标 ε_b=1,实为 **10**,V2-02.1) |
| `stoP` | 0.885419 | **100** | 10 | `[std-ref 未复核]` Wang 2013 世系(文献 ε_b=10;deck 实为 100,V2-02.1) |
| `btoBC`(存档) | 0.0885419 | **10** | 1 | `[DIAL §13]` 借自 bto298 |
| `btoL8`(存档) | 0.0885419 | **10** | 1 | `[DIAL §13]` 借自 bto298 |
| `pztG`(存档) | 0.0885419 | **10** | 1 | `[DIAL §13]` Ferret 原值(旧标 ε_b=1,实为 **10**,V2-02.1) |
| `pztH`(存档) | 0.0885419 | **10** | 1 | `[DIAL §13]` Ferret 原值(旧标 ε_b=1,实为 **10**,V2-02.1) |

**无 `permittivity` 的甲板:** `pmn0`(存档)、`pmnPT`(存档)、`pto`(存档)、`pzt06`(存档) —— 占位必填(普查 T6 #6)。

### K.6 弹性刚度(deck 数值即 GPa 数)

键名有两个变体:R1 schema 的 `C11/C12/C44`,与 Pan2019 系甲板的旧名 `c1111/c1122/c1212`。立方刚度**没有** Voigt-vs-张量歧义(c44 ≡ C44)。

| 甲板 | C11 | C12 | C44 | 键名变体 | 源级 |
|---|---|---|---|---|---|
| `bfoP` | 302 | 162 | 68 | `c1111/c1122/c1212` | `[pdf]` Zhang 2007(该句自称 first-principles、无引用号) |
| `bto118` | 275 | 179 | 54.3 | `C11/C12/C44` | `[toml-meta]` Ferret/HM2006 甲板 |
| `bto298` | 275 | 179 | 54.3 | `C11/C12/C44` | `[toml-meta]` Ferret/HM2006 甲板 |
| `btoLp` | 178 | 96.4 | 122 | `C11/C12/C44` | `[std-ref 未复核]` Li & Chen APL 88 Ref.18 脚注(论文自己的力学数) |
| `btoP` | 275 | 179 | 54.3 | `c1111/c1122/c1212` | `[pdf]`+`[LIT]` Sluka 等 Nat. Commun. 3, 748 (2012) Table 1;键名 `c1111/c1122/c1212` |
| `nnbm` | 156 | 96 | 127 | `C11/C12/C44` | `[pdf]`+`[LIT]` Wang 2026 SI(**存档不用**,elastic OFF) |
| `pbzoK` | 168.091 | 82.6211 | 40.6504 | `C11/C12/C44` | `[toml-meta]` Lum 2022 Table 1 |
| `pbzoT` | 260 | 120 | 12.7 | `C11/C12/C44` | `[pdf]`+`[LIT]` Yang 2025 SI(**存档不用**;论文自身用法未确认,T6 #23) |
| `stoA` | 315.6 | 101 | 119 | `C11/C12/C44` | `[std-ref 未复核]` Rabe App A Table 3 |
| `stoP` | 315.6 | 101 | 119 | `c1111/c1122/c1212` | `[pdf]` Pan 2019 SM;键名 `c1111/c1122/c1212` |
| `btoBC`(存档) | 275 | 179 | 54.3 | `C11/C12/C44` | `[toml-meta]` 借自 bto298 |
| `btoL8`(存档) | 275 | 179 | 54.3 | `C11/C12/C44` | `[toml-meta]` 借自 bto298 |
| `pto`(存档) | 174.603 | 79.3651 | 111.111 | `C11/C12/C44` | `[std-ref 未复核]` Rabe App A(柔度经 `units.cubic_compliance_to_stiffness` 反演) |
| `pzt06`(存档) | 169.591 | 81.8713 | 47.1698 | `C11/C12/C44` | `[std-ref 未复核]` Rabe App A Table 4 柔度反演 |

**整节缺失**(没有 `[elastic]` 或该段为空)的甲板:`pmn0`(存档)、`pmnD`(存档)、`pmnPT`(存档)、`pztG`(存档)、`pztH`(存档)(普查 T6 #7/#8/#9)。

### K.7 电致伸缩 Q(SI == deck,不换算,SPEC §9.1(a))

**三套键名并存**,普查 T6 #22 判定为“活的不一致”:`Q*_phys`(本仓库的 Voigt 风味,`elastic.py:93` `e[0,1]=0.5*Q44*P1*P2`)、`Q*_voigt`(明写记法)、`Q*_tensor`(全张量,`Q44_voigt = 4·Q1212_tensor`,SPEC §37.5 的命名陷阱)。

| 甲板 | Q11 | Q12 | Q44 | 键名变体 | 源级 |
|---|---|---|---|---|---|
| `bfoP` | 0.032 | −0.016 | 0.01 | `Q1111_tensor/Q1122_tensor/Q1212_tensor` | `[pdf]` Zhang 2007,**张量记法**(SPEC §37.5) |
| `bto118` | 0.11 | −0.045 | 0.029 | `Q11_phys/Q12_phys/Q44_phys` | `[toml-meta]` Ferret/HM2006 甲板(Voigt 风味 `Q*_phys`) |
| `bto298` | 0.11 | −0.045 | 0.029 | `Q11_phys/Q12_phys/Q44_phys` | `[toml-meta]` Ferret/HM2006 甲板 |
| `btoLp` | 0.1 | −0.034 | 0.029 | `Q11_phys/Q12_phys/Q44_phys` | `[std-ref 未复核]` Li & Chen APL 88 Ref.18 脚注(与 C 同一脚注) |
| `btoP` | 0.1104 | −0.0452 | 0.0578 | `Q11_voigt/Q12_voigt/Q44_voigt` | `[pdf]`+`[LIT]` Sluka 2012 Table 1,**Voigt 记法**(SPEC §37.5 裁定 R-c) |
| `nnbm` | 0.048 | −0.015 | 0.047 | `Q11_phys/Q12_phys/Q44_phys` | `[pdf]`+`[LIT]` Wang 2026 SI(**存档不用**) |
| `pbzoK` | 0.0726017 | −0.0270792 | 0.0314657 | `Q11_phys/Q12_phys/Q44_phys` | `[toml-meta]` Lum 2022 Table 1 |
| `pbzoT` | 0.06 | −0.015 | 0.0094 | `Q11_phys/Q12_phys/Q44_phys` | `[pdf]`+`[LIT]` Yang 2025 SI(**存档不用**) |
| `stoA` | 0.0457 | −0.0135 | 0.00957 | `Q11_phys/Q12_phys/Q44_phys` | `[std-ref 未复核]` Rabe App A Table 3 |
| `stoP` | 0.0457 | −0.0135 | 0.00957 | `Q11_voigt/Q12_voigt/Q44_voigt` | `[pdf]` Pan 2019 SM,**Voigt 记法** |
| `btoBC`(存档) | 0.11 | −0.045 | 0.029 | `Q11_phys/Q12_phys/Q44_phys` | `[toml-meta]` 借自 bto298 |
| `btoL8`(存档) | 0.11 | −0.045 | 0.029 | `Q11_phys/Q12_phys/Q44_phys` | `[toml-meta]` 借自 bto298 |
| `pto`(存档) | 0.089 | −0.026 | 0.0675 | `Q11_phys/Q12_phys/Q44_phys` | `[std-ref 未复核]` Rabe App A Table 5 |
| `pzt06`(存档) | 0.0811609 | −0.0294978 | 0.0335514 | `Q11_phys/Q12_phys/Q44_phys` | `[std-ref 未复核]` Rabe App A;**Q44 带源自身的 ½ 前因子**(见表下注) |

**整节缺失**的甲板:`pmn0`(存档)、`pmnD`(存档)、`pmnPT`(存档)、`pztG`(存档)、`pztH`(存档)。

**`pzt06` 的 ½ 前因子**(逐字取自 `params/archive/pzt_Haun_x0.6.toml` 的 `[meta].note`):“Q44 carries the source's own explicit 1/2 prefactor (kept, not symmetrized with Q11/Q12)”。

### K.8 AFE `q` 扇区(SPEC §23.1)

| 甲板 / 段 | 值 | 源级 |
|---|---|---|
| `pbzoK` `[landau_q]` | `aq1`=−0.0975、`aq11`=−0.25、`aq12`=0、`aq111`=2.66667、`aq112`=0、`aq123`=0 | `[toml-meta]` Lum 2022 Table 1 的符号变换(Pa=P+q);`Gq` 为 `[DIAL §23.5(d)]` 借用 |
| `pbzoK` `[coupling_pq]` | `t1`=−1.5、`t2`=0、`u1`=40、`u2`=40 | 四阶 `t1/t2` + 六阶 `u1/u2`,变换闭合式(SPEC §23.5(a)) |
| `pbzoK` `[gradient_q]` | `Gq`=1 | `[DIAL §23.5(d)]` 借 BTO 的 `G110`=1.0(它本身也不是测量值,T6 #1/#4) |
| `pbzoK` `[eigenstrain_q]` | `Lam11`=0.0203、`Lam12`=−0.00757155、`Lam44`=0.00879853 | `[toml-meta]` 各向同性锚点 Q(PbZrO₃)=2.03e−2 m⁴/C²;仅此一套(T6 #14) |

### K.9 八面体倾转 `θ` 扇区(SPEC §28.1)

**θ 的单位约定是逐甲板的**(SPEC §28.5 R2 / §28.7 R2):米模式(线位移,`si_to_deck_theta_landau`)与度模式(角度,`si_to_deck_landau`)的 `b11` 相差十个数量级以上 —— 肉眼“不合理”的那个可能才是对的。`b111/b112/b123` 在三套甲板里全为 0(θ 势只到四阶)。

| 甲板 | b1 | b11 | b12 | t11 | t12 | t44 | Gθ | Λ11^θ/Λ12^θ/Λ44^θ | 源级 |
|---|---|---|---|---|---|---|---|---|---|
| `stoA` | 142.56 | 169000 | 388000 | −174 | −75.5 | 585 | 1 | 8.7 / −7.8 / −9.2 | `[std-ref 未复核]` Rabe App A Table 3,θ 为**线位移(米模式)**;Λ^θ 按 §28.5 R6 **停放** |
| `pztG`(存档) | −17.0001 | 169000 | 388000 | −174 | −75.5 | 585 | 1 | — | **借用**:θ 与 (t11,t12,t44) 逐字取自 stoA 的 STO 生成式 @ 50 K —— 不是文献 PZT 倾转甲板 |
| `pztH`(存档) | 0.0003395 | 5.0791e−06 | 1.16609e−05 | −0.00430182 | −0.00186659 | 0.014463 | 1 | — | `[pdf]`+`[LIT]` Haun Ferroelectrics 99 Table —— θ 为**角度(度)**,逐甲板约定 |

`Gθ` 全部 = 1.0,与 `Gq` 同因:**没有任何来源发表过 θ 梯度系数**(`[DIAL §28.5]`,T6 #4)。Λ^θ 只有 `stoA` 有,且按 §28.5 R6 **停放**(T6 #14)。

### K.10 四阶梯度 `H11`(SPEC §32)

| 甲板 | H11 | 源级 |
|---|---|---|
| `nnbm` | 22.046 | `[DIAL §32]` 与 G110 联合标定于 λ\*≈51.2 nm;无任何已发表值 |

全仓库只有一套。`units.py` 无对应换算函数 —— 甲板里直接是 deck 数。

<!-- param_tables:end -->

---

## 附录:仓库内无来源的参数(普查 T6 全表)

`reports/archive/v1/REVIEW_wp30_param_census.md` T6 的 24 条,逐条给出**缺什么**与**本书在哪儿说它**。
SPEC §42:"a parameter whose range nobody knows says so" —— 下面每一条在正文里都写着
"仓库内无来源",没有一条被从记忆里补上。

| T6 # | 参数 | 缺的是什么 | 本书讨论处 |
|---|---|---|---|
| 1 | 梯度系数 `G11`/`G12`/`G44`/`G44p` 的一般范围 | 仓库里唯一有出处的梯度数是 Pan 2019 的 G₁₁₀=7.04e−11 SI;`G110=1.0` 是**重整化参考值**(`units.py:54-58` 自陈),不是测量值 | §C C.1 `G110` · C.3 #1 · §K 表 K.4 |
| 2 | BTO 比值组 0.51/−0.02/0.02/0 用于 PTO/PZT/PbZrO₃/STO | 七套非 BTO 甲板是**借**的,每套 `[meta].note` 自陈 "not X-validated" | §C C.1 比值条 · C.3 #2 · §K 表 K.4 |
| 3 | 四阶梯度 `H11` | 没有任何已发表值或区间;`nnbm` 的 22.046 与 `G110` 联合标定于 λ\*≈51.2 nm | §C C.1 `H11` · C.3 #3 · §K 表 K.10 |
| 4 | `Gq` 与 `Gtheta` | 没有任何来源发表过 q 或 θ 的梯度系数;两者都 = 1.0(借 BTO 那个本身也不是测量值的 `G110`) | §H `Gq` · §I `Gtheta` · §C C.3 #4 |
| 5 | PbZrO₃/Yang 2025 的 `kappa_b` = 50 | 主文只**符号性**定义 K,从未给数;50 是括号 [25,100] 内的旋钮 | §D D.1 κ 世系表 · §H `kappa_b` · §J `build_case` |
| 6 | `pto` 与 `pzt06` 的 κ(`permittivity`) | Rabe App A 不给背景介电常数,两套甲板标 PLACEHOLDER REQUIRED | §D D.1 · §E E.3 缺口 · §K 表 K.5 |
| 7 | `pztG`/`pztH` 的 C 与 Q | 两套 θ 车道 PZT 甲板**根本没有** `[elastic]`/`[electrostriction]` 段 | §E E.3 缺口 · §I 末尾 · §K 表 K.6/K.7 |
| 8 | PMN-PT 0.30 整套甲板 | 每个扇区都空;PRB 96, 054109 (2017) 引用已确认、数从未转录(付费墙) | §B B.3 #5 · §K 表 K.2 |
| 9 | PMN 的 `alpha11` 及所有高阶 | 故意**缺失**(不是 0);唯一的数是 `pmn_field.py:508` 的量级括号 (−0.3, 0.3),其注释自陈 "NOT PMN-derived" | §B B.1 `alpha11` 陷阱 · B.3 #4 · §K 表 K.2 |
| 10 | PMN 背景介电常数 `eps_r` | 只有经验排除法得到的括号 [500, 1000, 2000];无 PMN 专属的背景介电常数来源 | §G G.1 `eps_r` |
| 11 | 迁移率 `L`,因而 deck 的**时间单位** | L=1 是约定,仓库里没有任何测量过的动力学系数 —— deck 时间无 SI 锚,`dt` 没有物理范围,只有数值验证出的范围 | §F F.0 `L` · F.1 |
| 12 | deck 电场因子 ×1e−9 | **由 V2-02 关闭(不再是缺口)**:该因子不是约定而是**导出量** —— `units.si_to_deck_landau` 的前提(a₁_deck = a₁_SI×1e-9)只有在 E_deck = E_SI×1e-9 下才自洽;函数 `units.si_to_deck_field` 与门 K5 `permittivity_convention`(Leg C)同时落地,`nnbm` 的 0.10 deck ↔ 1e8 V/m 仍是在案交叉核对 | §A A.4 坑 3 · §D `E_ext` · §G G.4 · §J `e_max` |
| 13 | 各通道的无序关联长度 ξ | 两篇 Wang 论文都不给关联长度(`nnbm` 的 ξ=λ\*/2 是旋钮);PMN 的 ξ_COR=2.5 nm 是 2–3 nm 区间的中点;`make_random_strain` 的 ξ 与 `defect_field` 的 rms/ξ 无任何来源 | §G G.2 `xi_region_nm` · G.4 · G.5 `xi_x` · G.7 `defect_field` |
| 14 | 转伸缩 `Lam*_theta` 与 q 本征应变 `Lam*` | 各只有一套甲板(`stoA` 的 Λ^θ 且已**停放**;`pbzoK` 的 Λ),其他材料无任何范围 | §H `Lam11/Lam12/Lam44` · §I `Lam*_theta` · §K 表 K.8/K.9 |
| 15 | 协议速率 `n_dwell`、`e_max`、`n1` | 没有任何有出处的实验场速率;回线速率是协议的自由参数,只能靠阶梯收敛定标 | §J `n1` · `e_max` · `n_dwell` |
| 16 | 复合几何 `x_AN`、`fractions`、`xi`、标签种子 | 逐算例的设计选择,无文献范围 | §G G.6 |
| 17 | 系综规模 / 种子数 | 没有任何收敛性研究说"一次无序平均需要多少种子";仓库里唯一的 `n_seeds=14` 出自用途完全不同的 `calibrate_fl_chi_bias` | §F F.2 `seed` · §G G.1 `seed` |
| 18 | `btoP` 的 `alpha111` = 5.52 | 机理已由 `reports/LIT_pan2019_forensics.md` 查清,**取舍从未做出** —— 本书并列呈现两侧,不替读者选 | §B B.1 `alpha111` 陷阱 · B.3 #1 · §K 表 K.2 |
| 19 | STO 的 α₁ 常数 35.5 K vs 36.5 K | Rabe App A 与 Pan 2019 的一处真实转录冲突,两侧均 300 dpi 确认;`stoA` 与 `stoP` 因此按构造不相等 | §B B.3 #2 · §K 表 K.2 |
| 20 | STO Table-3 的逐行归属 | Rabe App A 对整表统一引 refs [10]–[15],任何单个 STO 系数都无法归到具体文章 | §B B.3 #3 · §I `b1`/`b11`/`b12` 陷阱 · §K 表 K.1 |
| 21 | `bto298` / `bto118` 的出处 | 权威是 **Ferret 甲板文件**不是论文 —— 仓库自己的基准甲板没有文献范围 | §B B.3 #6 · §K 表 K.1 |
| 22 | Q 的三套命名与 `pzt06` 的 ½ 前因子 | `Q*_phys`/`Q*_voigt`/`Q*_tensor` 并存,`pzt06` 的 Q44 带源自身的 ½ —— §37.5 的 Voigt-vs-张量陷阱是**活的不一致**,不是一个已定的范围 | §E E.1 "Q 的三套命名" · §G G.7 · §K 表 K.7 |
| 23 | Yang 2025 自身的弹性协议 | C/Q 逐字停放,但论文怎么用它们未确认 | §E E.3 缺口 · §K 表 K.6 |
| 24 | PZT 的 R3m/R3c 相界 | Corker 1998 不在手(`literature/README.md` "Wanted" 第 4 优先),两套 R3c 甲板的 (x, T) 论证未决 | §I 末尾"θ 通道的未决前提" |
