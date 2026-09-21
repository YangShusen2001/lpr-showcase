# LPRNet accuracy acceptance on real third-party data

- Source: `sirius-ai/LPRNet_Pytorch` `data/test/` -- real photographs,
  filenames are the human ground truth, no synthetic data.
- This is LPRNet's *own* test split: the comparison is biased in
  LPRNet's favour, and rpv3 was never trained on it.
- n = 1000 crops
- LPRNet preprocessing: official recipe (stretch 94x24, BGR, (x-127.5)/128)
- LPRNet dictionary: official `CHARS` from the upstream repo (68 entries, blank=67)

| recogniser | correct | accuracy |
|---|---|---|
| LPRNet | 888/1000 | 88.8% |
| rpv3 (production) | 906/1000 | 90.6% |

Paired McNemar: LPRNet-only-correct = 71, rpv3-only-correct = 89, exact p = 0.1788

Error profile (recomputed row-by-row from the tables below, not eyeballed):
rpv3 94 errors = 84 same-length substitutions + 10 length errors (1.0%);
LPRNet 112 errors = 42 same-length + 70 length errors (7.0%).

> Reproducibility: re-run on the on-disk dataset (`_evidence/A16-verify-rerun.log`)
> reproduces 888/1000, 906/1000 and p=0.1788 exactly.

## LPRNet errors

| file | gt | LPRNet | rpv3 |
|---|---|---|---|
| 京PL3N67.jpg | 京PL3N67 | <0>PL3N67 | 京PL3N67 |
| 川JK0707.jpg | 川JK0707 | 渝JK0707 | 川JK0707 |
| 川X90621.jpg | 川X90621 | 渝X90621 | 川X90621 |
| 沪AMS087.jpg | 沪AMS087 | 京AMS087 | 沪AMS087 |
| 沪C21F13.jpg | 沪C21F13 | 京C21F13 | 沪C21F13 |
| 沪C8GK31.jpg | 沪C8GK31 | 京C8GK31 | 沪C8GK31 |
| 浙D911SY.jpg | 浙D911SY | 浙D91186Y | 浙D911SY |
| 浙D9594H.jpg | 浙D9594H | 浙D9594HM | 浙D9594H |
| 皖A051N0.jpg | 皖A051N0 | 皖A51N0 | 皖A051N0 |
| 皖A0G826.jpg | 皖A0G826 | 皖A0G6826 | 皖A0G826 |
| 皖A0W669.jpg | 皖A0W669 | 皖A0W69 | 皖A0W669 |
| 皖A1X751.jpg | 皖A1X751 | 津AWM73 | 皖A1X751 |
| 皖A32C32.jpg | 皖A32C32 | 皖A32C362 | 皖A32C32 |
| 皖A3Y127.jpg | 皖A3Y127 | 皖A3127 | 皖A3Y127 |
| 皖A506T7.jpg | 皖A506T7 | 皖AJ6T7 | 皖A506T7 |
| 皖A531J3.jpg | 皖A531J3 | 皖A5331J3 | 皖A531J3 |
| 皖A586H7.jpg | 皖A586H7 | 皖A586PP7 | 粤A586H7 |
| 皖A59D31.jpg | 皖A59D31 | 皖A59031 | 皖A59D31 |
| 皖A5G895.jpg | 皖A5G895 | 皖A5G08985 | 粤A5G422 |
| 皖A5S422.jpg | 皖A5S422 | 皖A55422 | 皖A5S422 |
| 皖A5S939.jpg | 皖A5S939 | 皖A559397 | 皖A5S939 |
| 皖A79Y11.jpg | 皖A79Y11 | 青A791 | 皖A79Y11 |
| 皖A84A12.jpg | 皖A84A12 | 皖AA84A12 | 皖A84A12 |
| 皖A88G02.jpg | 皖A88G02 | 皖A88G091 | A86603 |
| 皖A88J99.jpg | 皖A88J99 | 皖A68J199 | 皖A88J99 |
| 皖A9U522.jpg | 皖A9U522 | 皖A9U5229 | 皖A9U522 |
| 皖AA9Z96.jpg | 皖AA9Z96 | 皖AA97Z96 | 皖AA9Z96 |
| 皖AB166M.jpg | 皖AB166M | 皖AG166M | 沪A3A656 |
| 皖AB921S.jpg | 皖AB921S | 皖AB921S5 | 皖AB921S |
| 皖ABB313.jpg | 皖ABB313 | 皖ABB315 | 皖ABB313 |
| 皖ABH733.jpg | 皖ABH733 | 皖AB733 | 皖ABH733 |
| 皖AC161T.jpg | 皖AC161T | 皖AC161T1 | 皖AC161T |
| 皖AC1758.jpg | 皖AC1758 | 皖AC117358 | 皖AC1758 |
| 皖AC585P.jpg | 皖AC585P | 皖AC585D | 皖AC585P |
| 皖AC705T.jpg | 皖AC705T | 皖AC705T1 | 皖AC705T |
| 皖AC828S.jpg | 皖AC828S | 皖AC8283A | 皖AC828S |
| 皖AC8J48.jpg | 皖AC8J48 | 皖AC8JA8 | 粤AD8J48 |
| 皖ACC998.jpg | 皖ACC998 | 皖AUCC998 | 皖ACC998 |
| 皖ACD429.jpg | 皖ACD429 | 皖AC0429 | 皖ACD429 |
| 皖ACD775.jpg | 皖ACD775 | 皖AC0775 | 皖ACD775 |
| 皖AE877C.jpg | 皖AE877C | 皖AE8770C | 皖AE877C |
| 皖AE9639.jpg | 皖AE9639 | 皖AE96369 | 皖AE9639 |
| 皖AG0D55.jpg | 皖AG0D55 | GAGDDB5B | 皖AG0D55 |
| 皖AGS527.jpg | 皖AGS527 | 皖A0GS527 | 皖AGS527 |
| 皖AH030P.jpg | 皖AH030P | 皖AH030B | 皖AH030P |
| 皖AH7F99.jpg | 皖AH7F99 | 藏A1 | 皖AH7F99 |
| 皖AH926B.jpg | 皖AH926B | 皖AH92Z6B | 桂AH926B |
| 皖AJ0J22.jpg | 皖AJ0J22 | 黑95 | E3B7 |
| 皖AJ7642.jpg | 皖AJ7642 | 皖AJ1642 | 皖AJ7642 |
| 皖AJ8F18.jpg | 皖AJ8F18 | 津AJ8F18 | 皖AJ8F18 |
| 皖AJ932K.jpg | 皖AJ932K | 皖AY932KV | 鄂N9932K |
| 皖AJ998V.jpg | 皖AJ998V | 皖AJ998N | 粤AJ998V |
| 皖AJU922.jpg | 皖AJU922 | 闽JU922 | 陕AJU922 |
| 皖AJV196.jpg | 皖AJV196 | 皖AJ1V196 | 皖AJV196 |
| 皖AJV866.jpg | 皖AJV866 | 皖AJ1V866 | 皖AJV866 |
| 皖AK383K.jpg | 皖AK383K | 青AK383K | 皖AK383K |
| 皖AK700C.jpg | 皖AK700C | 皖AK7000 | 皖AK700C |
| 皖AKS189.jpg | 皖AKS189 | 皖AAKS189 | 皖AKS189 |
| 皖AKW105.jpg | 皖AKW105 | 皖AKW106 | 赣AKW165 |
| 皖AL136E.jpg | 皖AL136E | 皖AL13G5 | 皖AL136E |
| 皖AL193R.jpg | 皖AL193R | 皖AL1393R | A173 |
| 皖AL5W29.jpg | 皖AL5W29 | 皖262X | 皖4229 |
| 皖ALN856.jpg | 皖ALN856 | 皖A1LN856 | 皖ALN856 |
| 皖ALR022.jpg | 皖ALR022 | 蒙ALRD39 | 皖ALR022 |
| 皖ALW592.jpg | 皖ALW592 | 皖ALW59 | 皖ALW592 |
| 皖AP666P.jpg | 皖AP666P | 皖8666P | R6560 |
| 皖AQQ677.jpg | 皖AQQ677 | 皖AQ0677 | 皖AQQ677 |
| 皖AR8X99.jpg | 皖AR8X99 | 皖ASSJY9 | 粤AR8X99 |
| 皖AS0825.jpg | 皖AS0825 | QAS082 | 云AS0825 |
| 皖AS522L.jpg | 皖AS522L | 皖AS5221 | 皖AS522L |
| 皖AS6X81.jpg | 皖AS6X81 | 皖A5S6AX81 | 皖AS6X81 |
| 皖AS836Z.jpg | 皖AS836Z | 皖AS8367 | 粤A3S8361 |
| 皖ASS990.jpg | 皖ASS990 | 皖ASS5990 | 皖ASS990 |
| 皖ASZ600.jpg | 皖ASZ600 | 皖AS7600 | 皖ASZ600 |
| 皖AT0D33.jpg | 皖AT0D33 | 皖AT00330 | 皖AT0D33 |
| 皖AT813C.jpg | 皖AT813C | 皖AT8130 | 皖AT813C |
| 皖ATC890.jpg | 皖ATC890 | 豫AJG83G | 皖ATC890 |
| 皖AU833R.jpg | 皖AU833R | 皖A833R | 京AU833R |
| 皖AUS393.jpg | 皖AUS393 | 皖AUGS393 | 皖AUS393 |
| 皖AV6T88.jpg | 皖AV6T88 | 皖AV6188 | 皖AV6T88 |
| 皖AVQ091.jpg | 皖AVQ091 | 皖AVQ0091 | 皖AVQ091 |
| 皖AVQ225.jpg | 皖AVQ225 | 皖AV0225 | 湘ANQ225 |
| 皖AW1A28.jpg | 皖AW1A28 | 皖AVW1A28 | 皖AW1A28 |
| 皖AW5V00.jpg | 皖AW5V00 | 皖AWV00 | 陕AW5V00 |
| 皖AX5716.jpg | 皖AX5716 | 皖A5716 | 皖AX5716 |
| 皖AX702L.jpg | 皖AX702L | 皖AX70217 | 皖AX702L |
| 皖AX788P.jpg | 皖AX788P | 皖AX788D | 皖AX788P |
| 皖AX799L.jpg | 皖AX799L | 皖AX799L1 | 皖AX7991 |
| 皖AY1V22.jpg | 皖AY1V22 | 皖A1V22 | 皖AY1V22 |
| 皖AY5L09.jpg | 皖AY5L09 | 津AGL09 | 皖AY5L09 |
| 皖AY605B.jpg | 皖AY605B | 皖AY605BH | 皖A605B |
| 皖AY710V.jpg | 皖AY710V | 津AY710V | 皖AY710V |
| 皖AYG248.jpg | 皖AYG248 | 皖AY6G248 | 皖AYG248 |
| 皖AZ1X26.jpg | 皖AZ1X26 | 皖A1X26 | 皖AZ1X26 |
| 皖AZ835P.jpg | 皖AZ835P | 皖AZ83B5P | 皖AZ835P |
| 皖AZ9D80.jpg | 皖AZ9D80 | 皖AZ79D80 | 皖AZ9D80 |
| 皖BU5521.jpg | 皖BU5521 | 皖1U5521 | 皖BU5521 |
| 皖DFX886.jpg | 皖DFX886 | 皖DFX882 | 皖DFX886 |
| 皖DJ9650.jpg | 皖DJ9650 | 赣D9650 | 湘DJ9650 |
| 皖N1B096.jpg | 皖N1B096 | 皖N1B092 | 皖N1B096 |
| 皖NXT689.jpg | 皖NXT689 | 皖NXT6897 | 皖NXT689 |
| 皖PB2153.jpg | 皖PB2153 | 皖PB2Z153 | 皖PB2153 |
| 粤A3ZB32.jpg | 粤A3ZB32 | 粤A3ZB832 | 粤A3ZB32 |
| 粤B0GJ76.jpg | 粤B0GJ76 | 鄂B0GJ76 | 粤B0GJ76 |
| 苏A57NT9.jpg | 苏A57NT9 | 苏A7NT9 | 苏A57NT9 |
| 苏A8DC31.jpg | 苏A8DC31 | 皖A8DC31 | 苏A8DC31 |
| 苏B810FT.jpg | 苏B810FT | WB810F | 苏B810FT |
| 苏G2F335.jpg | 苏G2F335 | 皖G2F335 | 苏G2F335 |
| 苏HXN335.jpg | 苏HXN335 | 苏HTXN335 | 苏HXN335 |
| 豫EZP772.jpg | 豫EZP772 | 皖EZP772 | 豫EZP772 |
| 豫RM6396.jpg | 豫RM6396 | 皖RM6396 | 豫RM6396 |
| 闽D33U29.jpg | 闽D33U29 | 皖D33U29 | 闽D33U29 |

## rpv3 errors

| file | gt | LPRNet | rpv3 |
|---|---|---|---|
| 皖A05279.jpg | 皖A05279 | 皖A05279 | 豫A05279 |
| 皖A09N61.jpg | 皖A09N61 | 皖A09N61 | 苏A09N31 |
| 皖A0H627.jpg | 皖A0H627 | 皖A0H627 | 京A0H627 |
| 皖A0V790.jpg | 皖A0V790 | 皖A0V790 | 皖A0N790 |
| 皖A0Y000.jpg | 皖A0Y000 | 皖A0Y000 | 粤A0Y000 |
| 皖A15C18.jpg | 皖A15C18 | 皖A15C18 | 京A15C18 |
| 皖A18145.jpg | 皖A18145 | 皖A18145 | 沪AJ8145 |
| 皖A25654.jpg | 皖A25654 | 皖A25654 | 陕A25654 |
| 皖A267A8.jpg | 皖A267A8 | 皖A267A8 | 桂A267A8 |
| 皖A29C52.jpg | 皖A29C52 | 皖A29C52 | 浙A29C52 |
| 皖A2U448.jpg | 皖A2U448 | 皖A2U448 | 粤A2J448 |
| 皖A34V88.jpg | 皖A34V88 | 皖A34V88 | 粤A34V88 |
| 皖A49P06.jpg | 皖A49P06 | 皖A49P06 | 京5A9P06 |
| 皖A4M405.jpg | 皖A4M405 | 皖A4M405 | 京A4M405 |
| 皖A586H7.jpg | 皖A586H7 | 皖A586PP7 | 粤A586H7 |
| 皖A596C7.jpg | 皖A596C7 | 皖A596C7 | 云A596C7 |
| 皖A5G895.jpg | 皖A5G895 | 皖A5G08985 | 粤A5G422 |
| 皖A5T949.jpg | 皖A5T949 | 皖A5T949 | 粤BA5T949 |
| 皖A625X0.jpg | 皖A625X0 | 皖A625X0 | 粤A625X0 |
| 皖A6V359.jpg | 皖A6V359 | 皖A6V359 | 桂A6V359 |
| 皖A75556.jpg | 皖A75556 | 皖A75556 | 粤A75556 |
| 皖A77Q40.jpg | 皖A77Q40 | 皖A77Q40 | 粤A77Q40 |
| 皖A84V80.jpg | 皖A84V80 | 皖A84V80 | 粤A84V80 |
| 皖A884B0.jpg | 皖A884B0 | 皖A884B0 | 冀A884B0 |
| 皖A88G02.jpg | 皖A88G02 | 皖A88G091 | A86603 |
| 皖A906G2.jpg | 皖A906G2 | 皖A906G2 | 桂A906G2 |
| 皖AA1R81.jpg | 皖AA1R81 | 皖AA1R81 | 苏AA1R81 |
| 皖AA603W.jpg | 皖AA603W | 皖AA603W | 京AA603K |
| 皖AB166M.jpg | 皖AB166M | 皖AG166M | 沪A3A656 |
| 皖AB7182.jpg | 皖AB7182 | 皖AB7182 | 粤AB7182 |
| 皖AC8J48.jpg | 皖AC8J48 | 皖AC8JA8 | 粤AD8J48 |
| 皖AD168D.jpg | 皖AD168D | 皖AD168D | 粤ADU680 |
| 皖ADE783.jpg | 皖ADE783 | 皖ADE783 | 皖A0E783 |
| 皖AE920X.jpg | 皖AE920X | 皖AE920X | 藏AE920X |
| 皖AE955B.jpg | 皖AE955B | 皖AE955B | 粤AE955B |
| 皖AEL010.jpg | 皖AEL010 | 皖AEL010 | 京AEL010 |
| 皖AF823X.jpg | 皖AF823X | 皖AF823X | 赣AF823X |
| 皖AGG512.jpg | 皖AGG512 | 皖AGG512 | 陕AGG512 |
| 皖AH0V77.jpg | 皖AH0V77 | 皖AH0V77 | 粤AH0V77 |
| 皖AH8642.jpg | 皖AH8642 | 皖AH8642 | 苏AH8642 |
| 皖AH886C.jpg | 皖AH886C | 皖AH886C | 豫AH886C |
| 皖AH926B.jpg | 皖AH926B | 皖AH92Z6B | 桂AH926B |
| 皖AHB011.jpg | 皖AHB011 | 皖AHB011 | 粤AHB011 |
| 皖AHN713.jpg | 皖AHN713 | 皖AHN713 | 皖AHP713 |
| 皖AHQ831.jpg | 皖AHQ831 | 皖AHQ831 | 粤AHQ831 |
| 皖AHX511.jpg | 皖AHX511 | 皖AHX511 | 皖AHX5714 |
| 皖AHX974.jpg | 皖AHX974 | 皖AHX974 | 京AHX974 |
| 皖AJ0J22.jpg | 皖AJ0J22 | 黑95 | E3B7 |
| 皖AJ201U.jpg | 皖AJ201U | 皖AJ201U | 桂AJZ011 |
| 皖AJ660Z.jpg | 皖AJ660Z | 皖AJ660Z | 京AJ660Z |
| 皖AJ932K.jpg | 皖AJ932K | 皖AY932KV | 鄂N9932K |
| 皖AJ998V.jpg | 皖AJ998V | 皖AJ998N | 粤AJ998V |
| 皖AJU215.jpg | 皖AJU215 | 皖AJU215 | 粤AJU215 |
| 皖AJU922.jpg | 皖AJU922 | 闽JU922 | 陕AJU922 |
| 皖AKK123.jpg | 皖AKK123 | 皖AKK123 | 粤AKK123 |
| 皖AKW105.jpg | 皖AKW105 | 皖AKW106 | 赣AKW165 |
| 皖AL193R.jpg | 皖AL193R | 皖AL1393R | A173 |
| 皖AL509L.jpg | 皖AL509L | 皖AL509L | 皖AL509L1 |
| 皖AL5W29.jpg | 皖AL5W29 | 皖262X | 皖4229 |
| 皖AL6W26.jpg | 皖AL6W26 | 皖AL6W26 | 京AL6W26 |
| 皖ALJ620.jpg | 皖ALJ620 | 皖ALJ620 | 京ALJ620 |
| 皖AM336V.jpg | 皖AM336V | 皖AM336V | 京AM336V |
| 皖AM700E.jpg | 皖AM700E | 皖AM700E | 云AM700E |
| 皖AN1336.jpg | 皖AN1336 | 皖AN1336 | 粤AU1336 |
| 皖AP198F.jpg | 皖AP198F | 皖AP198F | 京AP198F |
| 皖AP666P.jpg | 皖AP666P | 皖8666P | R6560 |
| 皖AP8950.jpg | 皖AP8950 | 皖AP8950 | 粤AP8950 |
| 皖AR5065.jpg | 皖AR5065 | 皖AR5065 | 新AR5065 |
| 皖AR690K.jpg | 皖AR690K | 皖AR690K | 京AR690K |
| 皖AR8X99.jpg | 皖AR8X99 | 皖ASSJY9 | 粤AR8X99 |
| 皖AS065M.jpg | 皖AS065M | 皖AS065M | 粤AS065M |
| 皖AS0825.jpg | 皖AS0825 | QAS082 | 云AS0825 |
| 皖AS371Z.jpg | 皖AS371Z | 皖AS371Z | 粤A33712 |
| 皖AS7E15.jpg | 皖AS7E15 | 皖AS7E15 | 琼AS7E15 |
| 皖AS836Z.jpg | 皖AS836Z | 皖AS8367 | 粤A3S8361 |
| 皖AS8L21.jpg | 皖AS8L21 | 皖AS8L21 | 粤AS8L21 |
| 皖AT152M.jpg | 皖AT152M | 皖AT152M | 沪AT152M |
| 皖ATB916.jpg | 皖ATB916 | 皖ATB916 | 京ATB916 |
| 皖AU833R.jpg | 皖AU833R | 皖A833R | 京AU833R |
| 皖AUU185.jpg | 皖AUU185 | 皖AUU185 | 粤AUU185 |
| 皖AVQ225.jpg | 皖AVQ225 | 皖AV0225 | 湘ANQ225 |
| 皖AW5V00.jpg | 皖AW5V00 | 皖AWV00 | 陕AW5V00 |
| 皖AWU124.jpg | 皖AWU124 | 皖AWU124 | 苏AWJ124 |
| 皖AX799L.jpg | 皖AX799L | 皖AX799L1 | 皖AX7991 |
| 皖AXC602.jpg | 皖AXC602 | 皖AXC602 | 京AXC602 |
| 皖AY3633.jpg | 皖AY3633 | 皖AY3633 | 琼AY3633 |
| 皖AY4K56.jpg | 皖AY4K56 | 皖AY4K56 | 陕AY4K56 |
| 皖AY605B.jpg | 皖AY605B | 皖AY605BH | 皖A605B |
| 皖AY628X.jpg | 皖AY628X | 皖AY628X | 鄂AX628X |
| 皖AY800C.jpg | 皖AY800C | 皖AY800C | 粤AX800C |
| 皖AZX040.jpg | 皖AZX040 | 皖AZX040 | 粤A7X040 |
| 皖DJ9650.jpg | 皖DJ9650 | 赣D9650 | 湘DJ9650 |
| 皖H18147.jpg | 皖H18147 | 皖H18147 | 皖HJ8147 |
| 苏E5YR23.jpg | 苏E5YR23 | 苏E5YR23 | 苏E5XR23 |
