# Ridge-to-exact W2 numerical audit

All certificates are computed per run before aggregation. The conservative value uses the reported Bonferroni simultaneous Gaussian/Wald confidence level, conditional on the fitted model and fixed evaluated environment set. Confidence is simultaneous within each run, not jointly across all seeds and methods in this report.

## Per-run test audits

| Dataset | Method | lambda_FG | Seed | Schur ridge | Conf. | min J_NN (LCB) | max gap (UCB) | ridge I | exact I | lower I | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CivilComments-WILDS | diagnostic | -- | 13 | 0.001 | 0.95 | 1.4575 (1.41844) | 0.00115727 (0.00132283) | 0.499076 | 0.498894 | 0.463662 | audited |
| CivilComments-WILDS | diagnostic | -- | 14 | 0.001 | 0.95 | 1.45594 (1.41696) | 0.00114778 (0.00131225) | 0.499788 | 0.499606 | 0.464371 | audited |
| CivilComments-WILDS | diagnostic | -- | 15 | 0.001 | 0.95 | 1.45673 (1.4177) | 0.00115951 (0.00132552) | 0.50008 | 0.499899 | 0.464564 | audited |
| CivilComments-WILDS | diagnostic | -- | 16 | 0.001 | 0.95 | 1.45697 (1.41794) | 0.00115563 (0.00132099) | 0.497216 | 0.497034 | 0.461847 | audited |
| CivilComments-WILDS | diagnostic | -- | 17 | 0.001 | 0.95 | 1.45686 (1.41783) | 0.00114735 (0.00131206) | 0.500221 | 0.500041 | 0.464796 | audited |
| Waterbirds via WILDS loader | diagnostic | -- | 17 | 0.001 | 0.95 | 0.211349 (0.178152) | 0.00125011 (0.00166114) | 0.16463 | 0.163919 | 0.15041 | audited |
| Waterbirds via WILDS loader | diagnostic | -- | 18 | 0.001 | 0.95 | 0.211349 (0.178152) | 0.00125011 (0.00166114) | 0.16463 | 0.163919 | 0.15041 | audited |
| Waterbirds via WILDS loader | diagnostic | -- | 19 | 0.001 | 0.95 | 0.211349 (0.178152) | 0.00125011 (0.00166114) | 0.16463 | 0.163919 | 0.15041 | audited |
| Waterbirds via WILDS loader | diagnostic | -- | 20 | 0.001 | 0.95 | 0.211349 (0.178152) | 0.00125011 (0.00166114) | 0.16463 | 0.163919 | 0.15041 | audited |
| Waterbirds via WILDS loader | diagnostic | -- | 21 | 0.001 | 0.95 | 0.211349 (0.178152) | 0.00125011 (0.00166114) | 0.16463 | 0.163919 | 0.15041 | audited |
| Waterbirds via WILDS loader | erm | 0 | 17 | 0.001 | 0.95 | 0.227201 (0.19172) | 0.00129802 (0.00173596) | 0.198234 | 0.197629 | 0.181829 | audited |
| Waterbirds via WILDS loader | group_dro | 0 | 17 | 0.001 | 0.95 | 0.227201 (0.19171) | 0.00143253 (0.00197935) | 0.233928 | 0.233384 | 0.214798 | audited |
| Waterbirds via WILDS loader | loss_cvar | 0 | 17 | 0.001 | 0.95 | 0.227201 (0.191695) | 0.00123536 (0.00166487) | 0.229126 | 0.22851 | 0.210452 | audited |
| Waterbirds via WILDS loader | jtt | 0 | 17 | 0.001 | 0.95 | 0.227201 (0.191619) | 0.00127363 (0.00171252) | 0.232594 | 0.231998 | 0.213533 | audited |
| Waterbirds via WILDS loader | fisher_glass | 0.03 | 17 | 0.001 | 0.95 | 0.227201 (0.191895) | 0.000819154 (0.0012372) | 0.403336 | 0.402969 | 0.369068 | audited |
| Waterbirds via WILDS loader | group_dro_fisher_glass | 0.3 | 17 | 0.001 | 0.95 | 0.227201 (0.191873) | 0.000457863 (0.000789371) | 0.641692 | 0.641532 | 0.581076 | audited |
| Waterbirds via WILDS loader | erm | 0 | 18 | 0.001 | 0.95 | 0.227201 (0.191714) | 0.00129984 (0.00173797) | 0.197763 | 0.197157 | 0.181409 | audited |
| Waterbirds via WILDS loader | group_dro | 0 | 18 | 0.001 | 0.95 | 0.227201 (0.191682) | 0.00142159 (0.00196402) | 0.231969 | 0.231421 | 0.213089 | audited |
| Waterbirds via WILDS loader | loss_cvar | 0 | 18 | 0.001 | 0.95 | 0.227201 (0.191671) | 0.00124381 (0.0016756) | 0.228136 | 0.227519 | 0.209599 | audited |
| Waterbirds via WILDS loader | jtt | 0 | 18 | 0.001 | 0.95 | 0.227201 (0.191617) | 0.00127611 (0.00171561) | 0.23235 | 0.231754 | 0.213288 | audited |
| Waterbirds via WILDS loader | fisher_glass | 0.03 | 18 | 0.001 | 0.95 | 0.227201 (0.191875) | 0.000821253 (0.0012365) | 0.40132 | 0.400956 | 0.367105 | audited |
| Waterbirds via WILDS loader | group_dro_fisher_glass | 0.1 | 18 | 0.001 | 0.95 | 0.227201 (0.191883) | 0.000743953 (0.00120606) | 0.513669 | 0.513414 | 0.46802 | audited |
| Waterbirds via WILDS loader | erm | 0 | 19 | 0.001 | 0.95 | 0.227201 (0.191717) | 0.00129506 (0.00173248) | 0.199217 | 0.198613 | 0.182739 | audited |
| Waterbirds via WILDS loader | group_dro | 0 | 19 | 0.001 | 0.95 | 0.227201 (0.191704) | 0.00140572 (0.00194198) | 0.230286 | 0.229728 | 0.211425 | audited |
| Waterbirds via WILDS loader | loss_cvar | 0 | 19 | 0.001 | 0.95 | 0.227201 (0.191668) | 0.00122579 (0.00165517) | 0.231338 | 0.230724 | 0.212513 | audited |
| Waterbirds via WILDS loader | jtt | 0 | 19 | 0.001 | 0.95 | 0.227201 (0.19162) | 0.00127235 (0.00171117) | 0.232826 | 0.232231 | 0.213737 | audited |
| Waterbirds via WILDS loader | fisher_glass | 0.03 | 19 | 0.001 | 0.95 | 0.227201 (0.191868) | 0.000873176 (0.00130334) | 0.392143 | 0.391768 | 0.359123 | audited |
| Waterbirds via WILDS loader | group_dro_fisher_glass | 0.03 | 19 | 0.001 | 0.95 | 0.227201 (0.191899) | 0.00111199 (0.00162923) | 0.34819 | 0.347781 | 0.319768 | audited |
| Waterbirds via WILDS loader | erm | 0 | 20 | 0.001 | 0.95 | 0.227201 (0.191713) | 0.00129912 (0.00173723) | 0.198204 | 0.197599 | 0.181805 | audited |
| Waterbirds via WILDS loader | group_dro | 0 | 20 | 0.001 | 0.95 | 0.227201 (0.19177) | 0.0014374 (0.00198801) | 0.234103 | 0.233552 | 0.215054 | audited |
| Waterbirds via WILDS loader | loss_cvar | 0 | 20 | 0.001 | 0.95 | 0.227201 (0.191634) | 0.00123889 (0.0016702) | 0.230214 | 0.229597 | 0.211522 | audited |
| Waterbirds via WILDS loader | jtt | 0 | 20 | 0.001 | 0.95 | 0.227201 (0.191597) | 0.00127634 (0.00171485) | 0.232093 | 0.231496 | 0.213054 | audited |
| Waterbirds via WILDS loader | fisher_glass | 0.03 | 20 | 0.001 | 0.95 | 0.227201 (0.191885) | 0.000814037 (0.00123122) | 0.404661 | 0.404297 | 0.370567 | audited |
| Waterbirds via WILDS loader | group_dro_fisher_glass | 0.01 | 20 | 0.001 | 0.95 | 0.227201 (0.191916) | 0.00118456 (0.00171484) | 0.311447 | 0.310994 | 0.286495 | audited |
| Waterbirds via WILDS loader | erm | 0 | 21 | 0.001 | 0.95 | 0.227201 (0.19171) | 0.00129871 (0.00173621) | 0.197979 | 0.197373 | 0.181593 | audited |
| Waterbirds via WILDS loader | group_dro | 0 | 21 | 0.001 | 0.95 | 0.227201 (0.191792) | 0.00142201 (0.00196773) | 0.235107 | 0.234564 | 0.215958 | audited |
| Waterbirds via WILDS loader | loss_cvar | 0 | 21 | 0.001 | 0.95 | 0.227201 (0.191676) | 0.00122351 (0.00165467) | 0.232644 | 0.23203 | 0.213693 | audited |
| Waterbirds via WILDS loader | jtt | 0 | 21 | 0.001 | 0.95 | 0.227201 (0.191623) | 0.00127472 (0.0017141) | 0.232886 | 0.232291 | 0.213789 | audited |
| Waterbirds via WILDS loader | fisher_glass | 0.1 | 21 | 0.001 | 0.95 | 0.227201 (0.19188) | 0.000610711 (0.000961627) | 0.52086 | 0.520568 | 0.468681 | audited |
| Waterbirds via WILDS loader | group_dro_fisher_glass | 0.3 | 21 | 0.001 | 0.95 | 0.227201 (0.191896) | 0.000462173 (0.000790553) | 0.631387 | 0.631228 | 0.570999 | audited |
| waterbirds | group_dro_fisher_glass | 0.01 | 17 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.000774914 (0.00110397) | 0.860642 | 0.860475 | 0.738457 | audited |
| waterbirds | group_dro_fisher_glass | 0.01 | 18 | 0.001 | 0.95 | 0.349184 (0.349184) | 7.60302e-06 (5.63317e-05) | 0.847533 | 0.847525 | 0.725334 | audited |
| waterbirds | group_dro_fisher_glass | 0.01 | 19 | 0.001 | 0.95 | 0.349184 (0.349184) | 4.21273e-05 (0.000144757) | 0.753818 | 0.753793 | 0.687789 | audited |
| waterbirds | group_dro_fisher_glass | 0.01 | 20 | 0.001 | 0.95 | 0.349184 (0.349184) | 3.59567e-05 (0.000135168) | 0.8944 | 0.894391 | 0.761551 | audited |
| waterbirds | group_dro_fisher_glass | 0.01 | 21 | 0.001 | 0.95 | 0.349184 (0.349184) | 6.42808e-05 (0.000173883) | 0.858829 | 0.858765 | 0.730624 | audited |
| waterbirds | group_dro_fisher_glass | 0.001 | 17 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.000120208 (0.000261839) | 0.870546 | 0.870426 | 0.725881 | audited |
| waterbirds | group_dro_fisher_glass | 0.001 | 18 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.000233104 (0.00044655) | 0.886781 | 0.886765 | 0.737685 | audited |
| waterbirds | group_dro_fisher_glass | 0.001 | 19 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.00140313 (0.00173869) | 0.58361 | 0.582207 | 0.480954 | audited |
| waterbirds | group_dro_fisher_glass | 0.001 | 20 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.000227959 (0.000438654) | 0.849567 | 0.84949 | 0.721387 | audited |
| waterbirds | group_dro_fisher_glass | 0.001 | 21 | 0.001 | 0.95 | 0.349184 (0.349184) | 8.41185e-06 (6.93951e-05) | 0.906284 | 0.906279 | 0.771772 | audited |
| waterbirds | group_dro_fisher_glass | 0.0001 | 17 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.00172383 (0.00207389) | 0.448156 | 0.447753 | 0.410632 | audited |
| waterbirds | group_dro_fisher_glass | 0.0001 | 18 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.00211624 (0.00253217) | 0.392478 | 0.392016 | 0.361866 | audited |
| waterbirds | group_dro_fisher_glass | 0.0001 | 19 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.00129208 (0.00164885) | 0.708753 | 0.707461 | 0.597188 | audited |
| waterbirds | group_dro_fisher_glass | 0.0001 | 20 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.00140225 (0.00173201) | 0.44127 | 0.44086 | 0.403481 | audited |
| waterbirds | group_dro_fisher_glass | 0.0001 | 21 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.000852216 (0.0011925) | 0.732807 | 0.732672 | 0.671969 | audited |
| waterbirds | group_dro_fisher_glass | 0.003 | 17 | 0.001 | 0.95 | 0.349184 (0.349184) | 5.7914e-05 (0.000167781) | 0.93541 | 0.935352 | 0.789432 | audited |
| waterbirds | group_dro_fisher_glass | 0.003 | 18 | 0.001 | 0.95 | 0.349184 (0.349184) | 2.89356e-05 (0.000122166) | 0.887482 | 0.887465 | 0.786312 | audited |
| waterbirds | group_dro_fisher_glass | 0.003 | 19 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.000233104 (0.000297961) | 0.685058 | 0.68491 | 0.630151 | audited |
| waterbirds | group_dro_fisher_glass | 0.003 | 20 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.000432434 (0.000681669) | 0.9188 | 0.918368 | 0.777623 | audited |
| waterbirds | group_dro_fisher_glass | 0.003 | 21 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.000171571 (0.000362742) | 0.902048 | 0.90201 | 0.802908 | audited |
| waterbirds | group_dro_fisher_glass | 0.0003 | 17 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.00192071 (0.00230883) | 0.457501 | 0.457114 | 0.420502 | audited |
| waterbirds | group_dro_fisher_glass | 0.0003 | 18 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.00105417 (0.00144841) | 0.820994 | 0.820912 | 0.709198 | audited |
| waterbirds | group_dro_fisher_glass | 0.0003 | 19 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.00163957 (0.00205233) | 0.743014 | 0.742842 | 0.630508 | audited |
| waterbirds | group_dro_fisher_glass | 0.0003 | 20 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.000566648 (0.000845181) | 0.76468 | 0.764552 | 0.692338 | audited |
| waterbirds | group_dro_fisher_glass | 0.0003 | 21 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.000197074 (0.000363132) | 0.812943 | 0.812846 | 0.687321 | audited |
| waterbirds | group_dro | 5e-05 | 17 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.00181591 (0.00219612) | 0.295059 | 0.294558 | 0.270472 | audited |
| waterbirds | group_dro_fisher_glass | 5e-05 | 17 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.00161322 (0.00196371) | 0.267567 | 0.267044 | 0.244534 | audited |
| waterbirds | group_dro | 5e-05 | 18 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.00142625 (0.00175535) | 0.219004 | 0.218458 | 0.200382 | audited |
| waterbirds | group_dro_fisher_glass | 5e-05 | 18 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.00252044 (0.00300185) | 0.489972 | 0.489591 | 0.449442 | audited |
| waterbirds | group_dro | 5e-05 | 19 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.000973556 (0.00120402) | 0.262536 | 0.262009 | 0.240726 | audited |
| waterbirds | group_dro_fisher_glass | 5e-05 | 19 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.00140845 (0.00171923) | 0.431314 | 0.43095 | 0.39484 | audited |
| waterbirds | group_dro | 5e-05 | 20 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.00161767 (0.00195906) | 0.261128 | 0.260593 | 0.238995 | audited |
| waterbirds | group_dro_fisher_glass | 5e-05 | 20 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.00213262 (0.00254489) | 0.329985 | 0.329492 | 0.302868 | audited |
| waterbirds | group_dro | 5e-05 | 21 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.001511 (0.00183886) | 0.262111 | 0.261556 | 0.240558 | audited |
| waterbirds | group_dro_fisher_glass | 5e-05 | 21 | 0.001 | 0.95 | 0.349184 (0.349184) | 0.00189197 (0.00226815) | 0.421021 | 0.420605 | 0.38537 | audited |
| Waterbirds via WILDS loader | erm | 0 | 17 | 0.01 | 0.95 | 0.227201 (0.19172) | 0.0124877 (0.0166922) | 0.203567 | 0.197629 | 0.181331 | audited |
| Waterbirds via WILDS loader | group_dro | 0 | 17 | 0.01 | 0.95 | 0.227201 (0.19171) | 0.0137817 (0.0190328) | 0.238723 | 0.233384 | 0.214252 | audited |
| Waterbirds via WILDS loader | loss_cvar | 0 | 17 | 0.01 | 0.95 | 0.227201 (0.191695) | 0.0118849 (0.0160109) | 0.234557 | 0.22851 | 0.209844 | audited |
| Waterbirds via WILDS loader | jtt | 0 | 17 | 0.01 | 0.95 | 0.227201 (0.191619) | 0.012253 (0.0164475) | 0.237837 | 0.231998 | 0.212966 | audited |
| Waterbirds via WILDS loader | fisher_glass | 0.1 | 17 | 0.01 | 0.95 | 0.227201 (0.191861) | 0.00588571 (0.00928248) | 0.531179 | 0.528335 | 0.475039 | audited |
| Waterbirds via WILDS loader | group_dro_fisher_glass | 0.01 | 17 | 0.01 | 0.95 | 0.227201 (0.191905) | 0.0113968 (0.0163801) | 0.308225 | 0.303775 | 0.279269 | audited |
| Waterbirds via WILDS loader | erm | 0 | 18 | 0.01 | 0.95 | 0.227201 (0.191714) | 0.0125053 (0.0167108) | 0.203101 | 0.197157 | 0.180901 | audited |
| Waterbirds via WILDS loader | group_dro | 0 | 18 | 0.01 | 0.95 | 0.227201 (0.191682) | 0.0136765 (0.0188839) | 0.236791 | 0.231421 | 0.212552 | audited |
| Waterbirds via WILDS loader | loss_cvar | 0 | 18 | 0.01 | 0.95 | 0.227201 (0.191671) | 0.0119662 (0.0161084) | 0.233573 | 0.227519 | 0.208984 | audited |
| Waterbirds via WILDS loader | jtt | 0 | 18 | 0.01 | 0.95 | 0.227201 (0.191617) | 0.0122769 (0.016475) | 0.237601 | 0.231754 | 0.212717 | audited |
| Waterbirds via WILDS loader | fisher_glass | 0.1 | 18 | 0.01 | 0.95 | 0.227201 (0.191869) | 0.00589737 (0.00927239) | 0.520724 | 0.517876 | 0.465617 | audited |
| Waterbirds via WILDS loader | group_dro_fisher_glass | 0.01 | 18 | 0.01 | 0.95 | 0.227201 (0.191911) | 0.0113429 (0.0164053) | 0.315339 | 0.310938 | 0.285932 | audited |
| Waterbirds via WILDS loader | erm | 0 | 19 | 0.01 | 0.95 | 0.227201 (0.191717) | 0.0124592 (0.0166599) | 0.204541 | 0.198613 | 0.182246 | audited |
| Waterbirds via WILDS loader | group_dro | 0 | 19 | 0.01 | 0.95 | 0.227201 (0.191704) | 0.0135238 (0.0186705) | 0.235206 | 0.229728 | 0.21086 | audited |
| Waterbirds via WILDS loader | loss_cvar | 0 | 19 | 0.01 | 0.95 | 0.227201 (0.191668) | 0.0117928 (0.0159127) | 0.236743 | 0.230724 | 0.211887 | audited |
| Waterbirds via WILDS loader | jtt | 0 | 19 | 0.01 | 0.95 | 0.227201 (0.19162) | 0.0122407 (0.0164341) | 0.238068 | 0.232231 | 0.213173 | audited |
| Waterbirds via WILDS loader | fisher_glass | 0.03 | 19 | 0.01 | 0.95 | 0.227201 (0.191822) | 0.00806551 (0.0120332) | 0.395119 | 0.391437 | 0.358228 | audited |
| Waterbirds via WILDS loader | group_dro_fisher_glass | 0.3 | 19 | 0.01 | 0.95 | 0.227201 (0.191866) | 0.00517155 (0.00874018) | 0.583125 | 0.581106 | 0.525224 | audited |
| Waterbirds via WILDS loader | erm | 0 | 20 | 0.01 | 0.95 | 0.227201 (0.191713) | 0.0124983 (0.016703) | 0.203538 | 0.197599 | 0.181299 | audited |
| Waterbirds via WILDS loader | group_dro | 0 | 20 | 0.01 | 0.95 | 0.227201 (0.19177) | 0.0138286 (0.0191165) | 0.238949 | 0.233552 | 0.214505 | audited |
| Waterbirds via WILDS loader | loss_cvar | 0 | 20 | 0.01 | 0.95 | 0.227201 (0.191634) | 0.0119188 (0.0160601) | 0.23565 | 0.229597 | 0.210883 | audited |
| Waterbirds via WILDS loader | jtt | 0 | 20 | 0.01 | 0.95 | 0.227201 (0.191597) | 0.0122791 (0.0164733) | 0.237351 | 0.231496 | 0.21248 | audited |
| Waterbirds via WILDS loader | fisher_glass | 0.03 | 20 | 0.01 | 0.95 | 0.227201 (0.191772) | 0.00791178 (0.0117982) | 0.394438 | 0.390697 | 0.357308 | audited |
| Waterbirds via WILDS loader | group_dro_fisher_glass | 0.3 | 20 | 0.01 | 0.95 | 0.227201 (0.191859) | 0.00444681 (0.00738237) | 0.649248 | 0.647725 | 0.585123 | audited |
| Waterbirds via WILDS loader | erm | 0 | 21 | 0.01 | 0.95 | 0.227201 (0.19171) | 0.0124943 (0.0166942) | 0.20332 | 0.197373 | 0.181089 | audited |
| Waterbirds via WILDS loader | group_dro | 0 | 21 | 0.01 | 0.95 | 0.227201 (0.191792) | 0.0136806 (0.0189249) | 0.239897 | 0.234564 | 0.215402 | audited |
| Waterbirds via WILDS loader | loss_cvar | 0 | 21 | 0.01 | 0.95 | 0.227201 (0.191676) | 0.0117709 (0.0159058) | 0.238052 | 0.23203 | 0.213115 | audited |
| Waterbirds via WILDS loader | jtt | 0 | 21 | 0.01 | 0.95 | 0.227201 (0.191623) | 0.0122635 (0.0164614) | 0.23813 | 0.232291 | 0.213222 | audited |
| Waterbirds via WILDS loader | fisher_glass | 0.01 | 21 | 0.01 | 0.95 | 0.227201 (0.191783) | 0.00961672 (0.0137718) | 0.320612 | 0.3162 | 0.290449 | audited |
| Waterbirds via WILDS loader | group_dro_fisher_glass | 0.01 | 21 | 0.01 | 0.95 | 0.227201 (0.191909) | 0.0113538 (0.0164142) | 0.315413 | 0.311016 | 0.285996 | audited |
| Waterbirds via WILDS loader | erm | 0 | 17 | 0.0001 | 0.95 | 0.227201 (0.19172) | 0.000130316 (0.000174284) | 0.19769 | 0.197629 | 0.181883 | audited |
| Waterbirds via WILDS loader | group_dro | 0 | 17 | 0.0001 | 0.95 | 0.227201 (0.19171) | 0.00014382 (0.000198739) | 0.233438 | 0.233384 | 0.214849 | audited |
| Waterbirds via WILDS loader | loss_cvar | 0 | 17 | 0.0001 | 0.95 | 0.227201 (0.191695) | 0.000124025 (0.000167179) | 0.228571 | 0.22851 | 0.210515 | audited |
| Waterbirds via WILDS loader | jtt | 0 | 17 | 0.0001 | 0.95 | 0.227201 (0.191619) | 0.000127867 (0.000171908) | 0.232058 | 0.231998 | 0.213604 | audited |
| Waterbirds via WILDS loader | fisher_glass | 0.03 | 17 | 0.0001 | 0.95 | 0.227201 (0.191812) | 8.25198e-05 (0.000123446) | 0.394007 | 0.393969 | 0.360772 | audited |
| Waterbirds via WILDS loader | group_dro_fisher_glass | 0.01 | 17 | 0.0001 | 0.95 | 0.227201 (0.191859) | 0.000114695 (0.000164874) | 0.298184 | 0.298137 | 0.274591 | audited |
| Waterbirds via WILDS loader | erm | 0 | 18 | 0.0001 | 0.95 | 0.227201 (0.191714) | 0.000130499 (0.000174483) | 0.197218 | 0.197157 | 0.181462 | audited |
| Waterbirds via WILDS loader | group_dro | 0 | 18 | 0.0001 | 0.95 | 0.227201 (0.191682) | 0.000142722 (0.0001972) | 0.231476 | 0.231421 | 0.213137 | audited |
| Waterbirds via WILDS loader | loss_cvar | 0 | 18 | 0.0001 | 0.95 | 0.227201 (0.191671) | 0.000124874 (0.000168267) | 0.227581 | 0.227519 | 0.209656 | audited |
| Waterbirds via WILDS loader | jtt | 0 | 18 | 0.0001 | 0.95 | 0.227201 (0.191617) | 0.000128116 (0.000172223) | 0.231814 | 0.231754 | 0.213359 | audited |
| Waterbirds via WILDS loader | fisher_glass | 0.03 | 18 | 0.0001 | 0.95 | 0.227201 (0.191758) | 8.35182e-05 (0.000124498) | 0.390395 | 0.390357 | 0.357401 | audited |
| Waterbirds via WILDS loader | group_dro_fisher_glass | 0.1 | 18 | 0.0001 | 0.95 | 0.227201 (0.19186) | 7.46202e-05 (0.000120106) | 0.500293 | 0.500265 | 0.455984 | audited |
| Waterbirds via WILDS loader | erm | 0 | 19 | 0.0001 | 0.95 | 0.227201 (0.191717) | 0.000130019 (0.000173941) | 0.198673 | 0.198613 | 0.182803 | audited |
| Waterbirds via WILDS loader | group_dro | 0 | 19 | 0.0001 | 0.95 | 0.227201 (0.191704) | 0.000141129 (0.000194998) | 0.229784 | 0.229728 | 0.211469 | audited |
| Waterbirds via WILDS loader | loss_cvar | 0 | 19 | 0.0001 | 0.95 | 0.227201 (0.191668) | 0.000123064 (0.000166175) | 0.230786 | 0.230724 | 0.212568 | audited |
| Waterbirds via WILDS loader | jtt | 0 | 19 | 0.0001 | 0.95 | 0.227201 (0.19162) | 0.000127739 (0.00017177) | 0.23229 | 0.232231 | 0.213811 | audited |
| Waterbirds via WILDS loader | fisher_glass | 0.03 | 19 | 0.0001 | 0.95 | 0.227201 (0.191851) | 8.55582e-05 (0.000128079) | 0.394428 | 0.394391 | 0.361646 | audited |
| Waterbirds via WILDS loader | group_dro_fisher_glass | 0.3 | 19 | 0.0001 | 0.95 | 0.227201 (0.191837) | 5.42226e-05 (9.44831e-05) | 0.604516 | 0.604498 | 0.547016 | audited |
| Waterbirds via WILDS loader | erm | 0 | 20 | 0.0001 | 0.95 | 0.227201 (0.191713) | 0.000130427 (0.000174409) | 0.197659 | 0.197599 | 0.18186 | audited |
| Waterbirds via WILDS loader | group_dro | 0 | 20 | 0.0001 | 0.95 | 0.227201 (0.19177) | 0.000144309 (0.000199624) | 0.233608 | 0.233552 | 0.215104 | audited |
| Waterbirds via WILDS loader | loss_cvar | 0 | 20 | 0.0001 | 0.95 | 0.227201 (0.191634) | 0.000124379 (0.000167718) | 0.229659 | 0.229597 | 0.211564 | audited |
| Waterbirds via WILDS loader | jtt | 0 | 20 | 0.0001 | 0.95 | 0.227201 (0.191597) | 0.000128139 (0.000172187) | 0.231556 | 0.231496 | 0.213121 | audited |
| Waterbirds via WILDS loader | fisher_glass | 0.03 | 20 | 0.0001 | 0.95 | 0.227201 (0.191837) | 8.3798e-05 (0.000125358) | 0.393387 | 0.393349 | 0.360467 | audited |
| Waterbirds via WILDS loader | group_dro_fisher_glass | 0.01 | 20 | 0.0001 | 0.95 | 0.227201 (0.191918) | 0.000116336 (0.000169094) | 0.315126 | 0.315081 | 0.290427 | audited |
| Waterbirds via WILDS loader | erm | 0 | 21 | 0.0001 | 0.95 | 0.227201 (0.19171) | 0.000130385 (0.000174307) | 0.197433 | 0.197373 | 0.181645 | audited |
| Waterbirds via WILDS loader | group_dro | 0 | 21 | 0.0001 | 0.95 | 0.227201 (0.191792) | 0.000142764 (0.000197592) | 0.234618 | 0.234564 | 0.215997 | audited |
| Waterbirds via WILDS loader | loss_cvar | 0 | 21 | 0.0001 | 0.95 | 0.227201 (0.191676) | 0.000122836 (0.000166122) | 0.232092 | 0.23203 | 0.213755 | audited |
| Waterbirds via WILDS loader | jtt | 0 | 21 | 0.0001 | 0.95 | 0.227201 (0.191623) | 0.000127976 (0.000172064) | 0.232351 | 0.232291 | 0.213855 | audited |
| Waterbirds via WILDS loader | fisher_glass | 0.1 | 21 | 0.0001 | 0.95 | 0.227201 (0.191881) | 5.89725e-05 (9.33783e-05) | 0.527071 | 0.527042 | 0.472332 | audited |
| Waterbirds via WILDS loader | group_dro_fisher_glass | 0.3 | 21 | 0.0001 | 0.95 | 0.227201 (0.191849) | 4.52577e-05 (8.17916e-05) | 0.642425 | 0.64241 | 0.580977 | audited |

## Across-seed summaries

Means are descriptive only; the minimum lower certificate is the fail-closed multi-run quantity.

| Dataset | Method | lambda_FG | n | Schur ridge | min J_NN (LCB) | max gap (UCB) | ridge mean | exact mean | exact min | lower mean | lower min | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CivilComments-WILDS | diagnostic | -- | 5 | 0.001 | 1.45594 (1.41696) | 0.00115951 (0.00132552) | 0.499276 | 0.499095 | 0.497034 | 0.463848 | 0.461847 | ['audited'] |
| Waterbirds via WILDS loader | erm | 0 | 5 | 0.0001 | 0.227201 (0.19171) | 0.000130499 (0.000174483) | 0.197735 | 0.197674 | 0.197157 | 0.181931 | 0.181462 | ['audited'] |
| Waterbirds via WILDS loader | erm | 0 | 5 | 0.001 | 0.227201 (0.19171) | 0.00129984 (0.00173797) | 0.19828 | 0.197674 | 0.197157 | 0.181875 | 0.181409 | ['audited'] |
| Waterbirds via WILDS loader | erm | 0 | 5 | 0.01 | 0.227201 (0.19171) | 0.0125053 (0.0167108) | 0.203614 | 0.197674 | 0.197157 | 0.181373 | 0.180901 | ['audited'] |
| Waterbirds via WILDS loader | fisher_glass | 0.01 | 1 | 0.01 | 0.227201 (0.191783) | 0.00961672 (0.0137718) | 0.320612 | 0.3162 | 0.3162 | 0.290449 | 0.290449 | ['audited'] |
| Waterbirds via WILDS loader | fisher_glass | 0.03 | 4 | 0.0001 | 0.227201 (0.191758) | 8.55582e-05 (0.000128079) | 0.393054 | 0.393016 | 0.390357 | 0.360071 | 0.357401 | ['audited'] |
| Waterbirds via WILDS loader | fisher_glass | 0.03 | 4 | 0.001 | 0.227201 (0.191868) | 0.000873176 (0.00130334) | 0.400365 | 0.399998 | 0.391768 | 0.366466 | 0.359123 | ['audited'] |
| Waterbirds via WILDS loader | fisher_glass | 0.03 | 2 | 0.01 | 0.227201 (0.191772) | 0.00806551 (0.0120332) | 0.394779 | 0.391067 | 0.390697 | 0.357768 | 0.357308 | ['audited'] |
| Waterbirds via WILDS loader | fisher_glass | 0.1 | 1 | 0.0001 | 0.227201 (0.191881) | 5.89725e-05 (9.33783e-05) | 0.527071 | 0.527042 | 0.527042 | 0.472332 | 0.472332 | ['audited'] |
| Waterbirds via WILDS loader | fisher_glass | 0.1 | 1 | 0.001 | 0.227201 (0.19188) | 0.000610711 (0.000961627) | 0.52086 | 0.520568 | 0.520568 | 0.468681 | 0.468681 | ['audited'] |
| Waterbirds via WILDS loader | fisher_glass | 0.1 | 2 | 0.01 | 0.227201 (0.191861) | 0.00589737 (0.00928248) | 0.525952 | 0.523106 | 0.517876 | 0.470328 | 0.465617 | ['audited'] |
| Waterbirds via WILDS loader | group_dro | 0 | 5 | 0.0001 | 0.227201 (0.191682) | 0.000144309 (0.000199624) | 0.232585 | 0.23253 | 0.229728 | 0.214111 | 0.211469 | ['audited'] |
| Waterbirds via WILDS loader | group_dro | 0 | 5 | 0.001 | 0.227201 (0.191682) | 0.0014374 (0.00198801) | 0.233079 | 0.23253 | 0.229728 | 0.214065 | 0.211425 | ['audited'] |
| Waterbirds via WILDS loader | group_dro | 0 | 5 | 0.01 | 0.227201 (0.191682) | 0.0138286 (0.0191165) | 0.237913 | 0.23253 | 0.229728 | 0.213515 | 0.21086 | ['audited'] |
| Waterbirds via WILDS loader | group_dro_fisher_glass | 0.01 | 2 | 0.0001 | 0.227201 (0.191859) | 0.000116336 (0.000169094) | 0.306655 | 0.306609 | 0.298137 | 0.282509 | 0.274591 | ['audited'] |
| Waterbirds via WILDS loader | group_dro_fisher_glass | 0.01 | 1 | 0.001 | 0.227201 (0.191916) | 0.00118456 (0.00171484) | 0.311447 | 0.310994 | 0.310994 | 0.286495 | 0.286495 | ['audited'] |
| Waterbirds via WILDS loader | group_dro_fisher_glass | 0.01 | 3 | 0.01 | 0.227201 (0.191905) | 0.0113968 (0.0164142) | 0.312993 | 0.308576 | 0.303775 | 0.283732 | 0.279269 | ['audited'] |
| Waterbirds via WILDS loader | group_dro_fisher_glass | 0.03 | 1 | 0.001 | 0.227201 (0.191899) | 0.00111199 (0.00162923) | 0.34819 | 0.347781 | 0.347781 | 0.319768 | 0.319768 | ['audited'] |
| Waterbirds via WILDS loader | group_dro_fisher_glass | 0.1 | 1 | 0.0001 | 0.227201 (0.19186) | 7.46202e-05 (0.000120106) | 0.500293 | 0.500265 | 0.500265 | 0.455984 | 0.455984 | ['audited'] |
| Waterbirds via WILDS loader | group_dro_fisher_glass | 0.1 | 1 | 0.001 | 0.227201 (0.191883) | 0.000743953 (0.00120606) | 0.513669 | 0.513414 | 0.513414 | 0.46802 | 0.46802 | ['audited'] |
| Waterbirds via WILDS loader | group_dro_fisher_glass | 0.3 | 2 | 0.0001 | 0.227201 (0.191837) | 5.42226e-05 (9.44831e-05) | 0.62347 | 0.623454 | 0.604498 | 0.563996 | 0.547016 | ['audited'] |
| Waterbirds via WILDS loader | group_dro_fisher_glass | 0.3 | 2 | 0.001 | 0.227201 (0.191873) | 0.000462173 (0.000790553) | 0.636539 | 0.63638 | 0.631228 | 0.576038 | 0.570999 | ['audited'] |
| Waterbirds via WILDS loader | group_dro_fisher_glass | 0.3 | 2 | 0.01 | 0.227201 (0.191859) | 0.00517155 (0.00874018) | 0.616186 | 0.614416 | 0.581106 | 0.555173 | 0.525224 | ['audited'] |
| Waterbirds via WILDS loader | jtt | 0 | 5 | 0.0001 | 0.227201 (0.191597) | 0.000128139 (0.000172223) | 0.232014 | 0.231954 | 0.231496 | 0.21355 | 0.213121 | ['audited'] |
| Waterbirds via WILDS loader | jtt | 0 | 5 | 0.001 | 0.227201 (0.191597) | 0.00127634 (0.00171561) | 0.23255 | 0.231954 | 0.231496 | 0.21348 | 0.213054 | ['audited'] |
| Waterbirds via WILDS loader | jtt | 0 | 5 | 0.01 | 0.227201 (0.191597) | 0.0122791 (0.016475) | 0.237797 | 0.231954 | 0.231496 | 0.212912 | 0.21248 | ['audited'] |
| Waterbirds via WILDS loader | loss_cvar | 0 | 5 | 0.0001 | 0.227201 (0.191634) | 0.000124874 (0.000168267) | 0.229738 | 0.229676 | 0.227519 | 0.211612 | 0.209656 | ['audited'] |
| Waterbirds via WILDS loader | loss_cvar | 0 | 5 | 0.001 | 0.227201 (0.191634) | 0.00124381 (0.0016756) | 0.230292 | 0.229676 | 0.227519 | 0.211556 | 0.209599 | ['audited'] |
| Waterbirds via WILDS loader | loss_cvar | 0 | 5 | 0.01 | 0.227201 (0.191634) | 0.0119662 (0.0161084) | 0.235715 | 0.229676 | 0.227519 | 0.210943 | 0.208984 | ['audited'] |
| Waterbirds via WILDS loader | diagnostic | -- | 5 | 0.001 | 0.211349 (0.178152) | 0.00125011 (0.00166114) | 0.16463 | 0.163919 | 0.163919 | 0.15041 | 0.15041 | ['audited'] |
| waterbirds | group_dro | 5e-05 | 5 | 0.001 | 0.349184 (0.349184) | 0.00181591 (0.00219612) | 0.259968 | 0.259435 | 0.218458 | 0.238227 | 0.200382 | ['audited'] |
| waterbirds | group_dro_fisher_glass | 0.0001 | 5 | 0.001 | 0.349184 (0.349184) | 0.00211624 (0.00253217) | 0.544693 | 0.544152 | 0.392016 | 0.489027 | 0.361866 | ['audited'] |
| waterbirds | group_dro_fisher_glass | 0.0003 | 5 | 0.001 | 0.349184 (0.349184) | 0.00192071 (0.00230883) | 0.719826 | 0.719653 | 0.457114 | 0.627973 | 0.420502 | ['audited'] |
| waterbirds | group_dro_fisher_glass | 0.001 | 5 | 0.001 | 0.349184 (0.349184) | 0.00140313 (0.00173869) | 0.819358 | 0.819033 | 0.582207 | 0.687536 | 0.480954 | ['audited'] |
| waterbirds | group_dro_fisher_glass | 0.003 | 5 | 0.001 | 0.349184 (0.349184) | 0.000432434 (0.000681669) | 0.86576 | 0.865621 | 0.68491 | 0.757285 | 0.630151 | ['audited'] |
| waterbirds | group_dro_fisher_glass | 0.01 | 5 | 0.001 | 0.349184 (0.349184) | 0.000774914 (0.00110397) | 0.843044 | 0.84299 | 0.753793 | 0.728751 | 0.687789 | ['audited'] |
| waterbirds | group_dro_fisher_glass | 5e-05 | 5 | 0.001 | 0.349184 (0.349184) | 0.00252044 (0.00300185) | 0.387972 | 0.387536 | 0.267044 | 0.355411 | 0.244534 | ['audited'] |
