# Changelog

## [0.7.0](https://github.com/CosmicFrontierLabs/coast-sim/compare/v0.6.0...v0.7.0) (2026-06-23)


### Features

* Add default plan output regression check ([#212](https://github.com/CosmicFrontierLabs/coast-sim/issues/212)) ([9eb749d](https://github.com/CosmicFrontierLabs/coast-sim/commit/9eb749d4d5377fc0342083fec4377ef6d96f2461))
* Plan constraint-safe ground station roll phases ([#210](https://github.com/CosmicFrontierLabs/coast-sim/issues/210)) ([9f22995](https://github.com/CosmicFrontierLabs/coast-sim/commit/9f22995b4f9f6648dda9abde24ef051840177a06))
* reward collection time during target selection ([#190](https://github.com/CosmicFrontierLabs/coast-sim/issues/190)) ([24e13ec](https://github.com/CosmicFrontierLabs/coast-sim/commit/24e13ecb12464ca0459383410b44e55032563acd))
* **visualization:** Add Spacecraft 3d model based on configuration parameters ([#122](https://github.com/CosmicFrontierLabs/coast-sim/issues/122)) ([a90d067](https://github.com/CosmicFrontierLabs/coast-sim/commit/a90d067ee01a535d1c3d49f297a69efe724af97f))


### Bug Fixes

* `PlanExecutionMismatchError` from plan/execution bookkeeping drift ([#192](https://github.com/CosmicFrontierLabs/coast-sim/issues/192)) ([ba281e4](https://github.com/CosmicFrontierLabs/coast-sim/commit/ba281e4b4e00e18ca541564d75af18f8fdea2169))
* cancel superseded ACS slew commands ([#144](https://github.com/CosmicFrontierLabs/coast-sim/issues/144)) ([ae57f2c](https://github.com/CosmicFrontierLabs/coast-sim/commit/ae57f2cc1d2fbcf46a231261b70087e6ccbaa395))
* clear self.ppt when terminating an immediately-constrained charging PPT ([#182](https://github.com/CosmicFrontierLabs/coast-sim/issues/182)) ([70715d2](https://github.com/CosmicFrontierLabs/coast-sim/commit/70715d2c6f64ec42acc00dc97bd33d2733efe134))
* **documentation:** catch up on updates to documentation ([#176](https://github.com/CosmicFrontierLabs/coast-sim/issues/176)) ([cb76c8b](https://github.com/CosmicFrontierLabs/coast-sim/commit/cb76c8bedcd35d35300fd78d6d730b325fb0e302))
* enforce constraint-safe idle holds ([#191](https://github.com/CosmicFrontierLabs/coast-sim/issues/191)) ([dc3987d](https://github.com/CosmicFrontierLabs/coast-sim/commit/dc3987d0617fd48c331eaef9aae467a507653440))
* Enforce full-mission attitude constraints ([#209](https://github.com/CosmicFrontierLabs/coast-sim/issues/209)) ([d008968](https://github.com/CosmicFrontierLabs/coast-sim/commit/d0089689956048719117b12b3dd4637d53f3a5f5))
* export GSP tracking attitude metadata ([#146](https://github.com/CosmicFrontierLabs/coast-sim/issues/146)) ([9315949](https://github.com/CosmicFrontierLabs/coast-sim/commit/9315949f7a4f23f280cf7c43868abeec09e56680))
* keep QueueDITL target state aligned after command churn ([#184](https://github.com/CosmicFrontierLabs/coast-sim/issues/184)) ([fa875db](https://github.com/CosmicFrontierLabs/coast-sim/commit/fa875db2d2a796b556e3b0aef3005aa4b04f31d5))
* log recharge-threshold interruptions as charging ([#213](https://github.com/CosmicFrontierLabs/coast-sim/issues/213)) ([a098b0c](https://github.com/CosmicFrontierLabs/coast-sim/commit/a098b0ca1309b54ae160b2670b81c8004292c50c))
* only guard science dispatch below battery floor ([#188](https://github.com/CosmicFrontierLabs/coast-sim/issues/188)) ([a2bc559](https://github.com/CosmicFrontierLabs/coast-sim/commit/a2bc5597939d1c22aa5a021bc9d830d1a7c41143))
* prevent stale plan entry slew sync ([#166](https://github.com/CosmicFrontierLabs/coast-sim/issues/166)) ([34d42be](https://github.com/CosmicFrontierLabs/coast-sim/commit/34d42be3c801b2825fad00cd5b1375c9d82d40e1))
* refine hard-keepout validation and fault management ([#194](https://github.com/CosmicFrontierLabs/coast-sim/issues/194)) ([a41b914](https://github.com/CosmicFrontierLabs/coast-sim/commit/a41b914e364084b2e79437f3cd1e14a063c3f2bd))
* retry equal-merit target selection ([#175](https://github.com/CosmicFrontierLabs/coast-sim/issues/175)) ([3d3130c](https://github.com/CosmicFrontierLabs/coast-sim/commit/3d3130ce0da9a06054cb31a9671a1c1403a771a7))
* sync plan slew metadata with ACS execution ([#145](https://github.com/CosmicFrontierLabs/coast-sim/issues/145)) ([47d88b3](https://github.com/CosmicFrontierLabs/coast-sim/commit/47d88b35f607d06b2f4d0153437f4a35793c27b6))
* track GSP contacts with configured antenna vector ([#177](https://github.com/CosmicFrontierLabs/coast-sim/issues/177)) ([9e7be74](https://github.com/CosmicFrontierLabs/coast-sim/commit/9e7be74bcb239baf0c0d93e610b2691cc660c57f))
* use cheap pass slew trigger estimates ([#198](https://github.com/CosmicFrontierLabs/coast-sim/issues/198)) ([c8c86fc](https://github.com/CosmicFrontierLabs/coast-sim/commit/c8c86fc954079c2e2960f3b13fcea4838730e739))
* validate executed science/contact activity before export ([#169](https://github.com/CosmicFrontierLabs/coast-sim/issues/169)) ([5a85efe](https://github.com/CosmicFrontierLabs/coast-sim/commit/5a85efe0ebaf84b309dc9eda055ab47afad6cdb3))
* validate exported plan entries against ACS execution ([#147](https://github.com/CosmicFrontierLabs/coast-sim/issues/147)) ([daf50ae](https://github.com/CosmicFrontierLabs/coast-sim/commit/daf50ae2f093a600e940559fbfecfd939d7094ef))


### Performance Improvements

* various queue ditl optimizations ([#211](https://github.com/CosmicFrontierLabs/coast-sim/issues/211)) ([ec89809](https://github.com/CosmicFrontierLabs/coast-sim/commit/ec89809c3c1f5d40522570cf67a3da0627678ba7))


### Documentation

* seed quick DITL example targets ([#185](https://github.com/CosmicFrontierLabs/coast-sim/issues/185)) ([e723806](https://github.com/CosmicFrontierLabs/coast-sim/commit/e7238061e214aa0ca5594b350d830e44f5446eef))

## [0.6.0](https://github.com/CosmicFrontierLabs/coast-sim/compare/v0.5.0...v0.6.0) (2026-05-17)


### Features

* represent commanded ground-station passes as plan activities ([#139](https://github.com/CosmicFrontierLabs/coast-sim/issues/139)) ([a3ae04b](https://github.com/CosmicFrontierLabs/coast-sim/commit/a3ae04b623944381a2e479ca9c87adab029be4ac))


### Bug Fixes

* render plotly timeline hover times ([#137](https://github.com/CosmicFrontierLabs/coast-sim/issues/137)) ([7fa6a1d](https://github.com/CosmicFrontierLabs/coast-sim/commit/7fa6a1d0c761caafb1114ebbb708e75bb3051601))

## [0.5.0](https://github.com/CosmicFrontierLabs/coast-sim/compare/v0.4.1...v0.5.0) (2026-05-14)


### Features

* **slew:** add alternate slew algorithms ([#125](https://github.com/CosmicFrontierLabs/coast-sim/issues/125)) ([5658dc6](https://github.com/CosmicFrontierLabs/coast-sim/commit/5658dc6e6cd880f28a53996c048086193f1582b4))


### Bug Fixes

* revert change to notebook ([49a5b29](https://github.com/CosmicFrontierLabs/coast-sim/commit/49a5b29211eb8e24fcb09902ee017aae1c9e6114))
* **roll:** fix bug where roll angle was changing during observation ([#134](https://github.com/CosmicFrontierLabs/coast-sim/issues/134)) ([936572c](https://github.com/CosmicFrontierLabs/coast-sim/commit/936572c116389d05d0268571d91eab271d68a269))
* **tests:** import refactoring ([#131](https://github.com/CosmicFrontierLabs/coast-sim/issues/131)) ([9f90a26](https://github.com/CosmicFrontierLabs/coast-sim/commit/9f90a2611407eb6dfa15f268859f9daf9874aff3))
* under-collected science plan entries ([#136](https://github.com/CosmicFrontierLabs/coast-sim/issues/136)) ([49a5b29](https://github.com/CosmicFrontierLabs/coast-sim/commit/49a5b29211eb8e24fcb09902ee017aae1c9e6114))

## [0.4.1](https://github.com/CosmicFrontierLabs/coast-sim/compare/v0.4.0...v0.4.1) (2026-04-29)


### Bug Fixes

* fixes to annotation of YAML, fix roll constraints bug ([#126](https://github.com/CosmicFrontierLabs/coast-sim/issues/126)) ([10536bb](https://github.com/CosmicFrontierLabs/coast-sim/commit/10536bb2e138659ed4079ad2342ecd4372cb26b9))

## [0.4.0](https://github.com/CosmicFrontierLabs/coast-sim/compare/v0.3.0...v0.4.0) (2026-04-24)


### Features

* add radiator support ([#107](https://github.com/CosmicFrontierLabs/coast-sim/issues/107)) ([7c8b8a4](https://github.com/CosmicFrontierLabs/coast-sim/commit/7c8b8a4ef3950a090c8110efd5d5d263c6675a78))
* add support for Telescope definition in payload ([#121](https://github.com/CosmicFrontierLabs/coast-sim/issues/121)) ([172d6f5](https://github.com/CosmicFrontierLabs/coast-sim/commit/172d6f534d37c24ead5df09e292f8551f3722f2c))


### Bug Fixes

* plotly visualization updates ([#114](https://github.com/CosmicFrontierLabs/coast-sim/issues/114)) ([3dd6f67](https://github.com/CosmicFrontierLabs/coast-sim/commit/3dd6f67f63c7c4818fed10f822ab437f73fec7c8))

## [0.3.0](https://github.com/CosmicFrontierLabs/coast-sim/compare/v0.2.2...v0.3.0) (2026-04-14)


### Features

* update in_constraint_batch api to match rust_ephem 0.8.0 ([#118](https://github.com/CosmicFrontierLabs/coast-sim/issues/118)) ([3d68eaa](https://github.com/CosmicFrontierLabs/coast-sim/commit/3d68eaadce11bb621da3f1961ce564dd27e8f5fc))


### Bug Fixes

* add orbit constraints to config ([#115](https://github.com/CosmicFrontierLabs/coast-sim/issues/115)) ([1be09d9](https://github.com/CosmicFrontierLabs/coast-sim/commit/1be09d93d10000e92ea59f0339fd7f92b41b340d))

## [0.2.2](https://github.com/CosmicFrontierLabs/coast-sim/compare/v0.2.1...v0.2.2) (2026-04-01)


### Bug Fixes

* Star tracker visibility bugs ([#111](https://github.com/CosmicFrontierLabs/coast-sim/issues/111)) ([439227b](https://github.com/CosmicFrontierLabs/coast-sim/commit/439227b53f2f831ae293b37dd3a54f2e852ebf4a))

## [0.2.1](https://github.com/CosmicFrontierLabs/coast-sim/compare/v0.2.0...v0.2.1) (2026-03-17)


### Bug Fixes

* bug in star tracker rendering and offsets with roll angle ([#108](https://github.com/CosmicFrontierLabs/coast-sim/issues/108)) ([fb837fd](https://github.com/CosmicFrontierLabs/coast-sim/commit/fb837fd6c7942ff5c349a13a4c2a57ad12bafc0f))

## [0.2.0](https://github.com/CosmicFrontierLabs/coast-sim/compare/v0.1.2...v0.2.0) (2026-03-10)


### Features

* add star tracker support ([#98](https://github.com/CosmicFrontierLabs/coast-sim/issues/98)) ([ee418d6](https://github.com/CosmicFrontierLabs/coast-sim/commit/ee418d6c721dded95825304d49eaaf7969d0043a))
* add support for reading/writing configuration in YAML ([#91](https://github.com/CosmicFrontierLabs/coast-sim/issues/91)) ([bfd05df](https://github.com/CosmicFrontierLabs/coast-sim/commit/bfd05df740c23cbc621f8f4c76212914485ec26e))
* fault management refactor ([#99](https://github.com/CosmicFrontierLabs/coast-sim/issues/99)) ([e2a3139](https://github.com/CosmicFrontierLabs/coast-sim/commit/e2a3139e7a176e575f2c4c8ad9dbc91777affd2e))
* improve support for spacecraft roll angles ([#94](https://github.com/CosmicFrontierLabs/coast-sim/issues/94)) ([16c041f](https://github.com/CosmicFrontierLabs/coast-sim/commit/16c041f6444c91b3d06e6ff9c1166cc45fdfa357))
* redo how solar panels are defined ([#95](https://github.com/CosmicFrontierLabs/coast-sim/issues/95)) ([67d1aba](https://github.com/CosmicFrontierLabs/coast-sim/commit/67d1abafbbfa290b8bcb011f86143d9ada09c0df))
* rename `Payload`'s `payload` attribute to `instruments` ([#96](https://github.com/CosmicFrontierLabs/coast-sim/issues/96)) ([0d9dfc1](https://github.com/CosmicFrontierLabs/coast-sim/commit/0d9dfc11d4c74118bed247e80f8c2dc4ede9647e))
* Serialize DITL generated plans to disk ([#104](https://github.com/CosmicFrontierLabs/coast-sim/issues/104)) ([1a4db33](https://github.com/CosmicFrontierLabs/coast-sim/commit/1a4db3398784bdb0856bca29ed6b372817b745f5))
* **targets:** Add slew distance weight to target selection scoring ([#70](https://github.com/CosmicFrontierLabs/coast-sim/issues/70)) ([d74803f](https://github.com/CosmicFrontierLabs/coast-sim/commit/d74803faca89edabebe4a8639bcae6c558ec7ba9))
* **visibility:** add instantaneous field of regard calculator ([#102](https://github.com/CosmicFrontierLabs/coast-sim/issues/102)) ([1345b49](https://github.com/CosmicFrontierLabs/coast-sim/commit/1345b49272b3feb41c015e4b021cbf3e5facb06a))


### Bug Fixes

* Add pass-aware scheduling to prevent observation/pass conflicts ([#82](https://github.com/CosmicFrontierLabs/coast-sim/issues/82)) ([5c49ee6](https://github.com/CosmicFrontierLabs/coast-sim/commit/5c49ee6927d739f4881fe87aee582cb53e0500d0))
* Address three misc bugs missed in previous updates ([#83](https://github.com/CosmicFrontierLabs/coast-sim/issues/83)) ([b1865f8](https://github.com/CosmicFrontierLabs/coast-sim/commit/b1865f8cb51e5409e0065a1051b95a0efe2a9b2a))
* update documentation for 0.2.0 release ([#105](https://github.com/CosmicFrontierLabs/coast-sim/issues/105)) ([b7d113a](https://github.com/CosmicFrontierLabs/coast-sim/commit/b7d113a241ca4f2708bc4ce14f5dc6f0354ecb95))
* **vector:** Use 180° threshold in roll_over_angle for shortest path ([#78](https://github.com/CosmicFrontierLabs/coast-sim/issues/78)) ([536df89](https://github.com/CosmicFrontierLabs/coast-sim/commit/536df890ef36e9e2902f199a92e161b39e02ada8))


### Performance Improvements

* optimize emergency charging pointing search ([#74](https://github.com/CosmicFrontierLabs/coast-sim/issues/74)) ([7c34ef4](https://github.com/CosmicFrontierLabs/coast-sim/commit/7c34ef4550f6a521fc9a29eb448747446c35fa7b))
* Use rust-ephem 0.3.0 direct array access for sun/moon/earth positions ([#72](https://github.com/CosmicFrontierLabs/coast-sim/issues/72)) ([73ce95a](https://github.com/CosmicFrontierLabs/coast-sim/commit/73ce95acf524782c855d968c1bc6111775bc2470))
* Vectorize solar panel illumination calculations ([#65](https://github.com/CosmicFrontierLabs/coast-sim/issues/65)) ([b741afd](https://github.com/CosmicFrontierLabs/coast-sim/commit/b741afd63b1328021550e56943bb99d87a7fc91d))

## [0.1.2](https://github.com/CosmicFrontierLabs/coast-sim/compare/v0.1.1...v0.1.2) (2025-12-18)


### Bug Fixes

* image in PyPI ([#59](https://github.com/CosmicFrontierLabs/coast-sim/issues/59)) ([022a853](https://github.com/CosmicFrontierLabs/coast-sim/commit/022a853299a755f3fc866a66cb9dd5c2dd506377))

## [0.1.1](https://github.com/CosmicFrontierLabs/coast-sim/compare/v0.1.0...v0.1.1) (2025-12-18)


### Bug Fixes

* issues with namespace and build errors ([#56](https://github.com/CosmicFrontierLabs/coast-sim/issues/56)) ([8a89a10](https://github.com/CosmicFrontierLabs/coast-sim/commit/8a89a10b0585c4551c5b4a1c5b1507ef29108f7b))
* **visualization:** improve skyplot ([#58](https://github.com/CosmicFrontierLabs/coast-sim/issues/58)) ([080952a](https://github.com/CosmicFrontierLabs/coast-sim/commit/080952a570501e8a343c08e02627c0148e2e5a2f))

## 0.1.0 (2025-12-18)


### Features

* Add release please and start doing releases to PyPI ([#49](https://github.com/CosmicFrontierLabs/coast-sim/issues/49)) ([35258c6](https://github.com/CosmicFrontierLabs/coast-sim/commit/35258c641a73709a28a2330f51bf0edd2c55e9d7))
* add safe mode indicator to timeline plot ([#26](https://github.com/CosmicFrontierLabs/coast-sim/issues/26)) ([e42f050](https://github.com/CosmicFrontierLabs/coast-sim/commit/e42f0500de86c435efa51b149a8dae4fc856dbe7))
* add some basic github workflows ([277c103](https://github.com/CosmicFrontierLabs/coast-sim/commit/277c103a87993a327bf55f48bfc491ae6cbb4b8f))
* first commit of conops simulator ([852b650](https://github.com/CosmicFrontierLabs/coast-sim/commit/852b650082b4501fd755c1d6be341efbedccd728))
* remove print statements in favor of logging ([#38](https://github.com/CosmicFrontierLabs/coast-sim/issues/38)) ([0a1108a](https://github.com/CosmicFrontierLabs/coast-sim/commit/0a1108a511bd21fb7d197b1cd48e3865b9c03f3b))


### Bug Fixes

* actions ([7d8d19c](https://github.com/CosmicFrontierLabs/coast-sim/commit/7d8d19c1067d9442bca7be058d58bed5966c2038))
* add release-please config ([#53](https://github.com/CosmicFrontierLabs/coast-sim/issues/53)) ([7f786dd](https://github.com/CosmicFrontierLabs/coast-sim/commit/7f786dd7f391bdb598ad2386658dcf55411203d3))
* don't self host tests ([d515df6](https://github.com/CosmicFrontierLabs/coast-sim/commit/d515df63348ea11985402afc8b54c6ffac1d9939))
* fix plotting of charging ppts in ditl timeline ([#30](https://github.com/CosmicFrontierLabs/coast-sim/issues/30)) ([81cc720](https://github.com/CosmicFrontierLabs/coast-sim/commit/81cc72050f9554a1defc1063dbb52a118074b71e))
* formatting issues ([d5b5e42](https://github.com/CosmicFrontierLabs/coast-sim/commit/d5b5e420905ecb819533228cd52ea577657ddca6))
* lint action ([cebc3e8](https://github.com/CosmicFrontierLabs/coast-sim/commit/cebc3e80c78b46b94d0a0d489879c859a9555794))
* random docs stuff ([#46](https://github.com/CosmicFrontierLabs/coast-sim/issues/46)) ([2fae72c](https://github.com/CosmicFrontierLabs/coast-sim/commit/2fae72c01a6ba7016ea190f02f52c747e20731f5))
* remove rust mention ([ece0813](https://github.com/CosmicFrontierLabs/coast-sim/commit/ece08133ffd305130f10d36a89f64203ef31723e))
* remove skyconstraints import ([f5d010c](https://github.com/CosmicFrontierLabs/coast-sim/commit/f5d010c3f32263a69f05249336ed663c536bec8d))
* tests and updated code ([1ff768a](https://github.com/CosmicFrontierLabs/coast-sim/commit/1ff768a6a6a442dcafab2e070e3108faba543ccb))
* updates to examples ([cfea209](https://github.com/CosmicFrontierLabs/coast-sim/commit/cfea209d9d20a2229daf42c76ede33760ff9f6cd))
* use uv instead of pip ([1c9a500](https://github.com/CosmicFrontierLabs/coast-sim/commit/1c9a50050b72679c0be96613001a9fa38a8cfb4e))


### Documentation

* add sphinx docs ([#43](https://github.com/CosmicFrontierLabs/coast-sim/issues/43)) ([159c692](https://github.com/CosmicFrontierLabs/coast-sim/commit/159c692a0c30e5e8899a4f9c4af9e955a4f78442))
