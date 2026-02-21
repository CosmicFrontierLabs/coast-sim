# Changelog

## [0.2.0](https://github.com/CosmicFrontierLabs/coast-sim/compare/v0.1.2...v0.2.0) (2026-02-21)


### Features

* add support for reading/writing configuration in YAML ([#91](https://github.com/CosmicFrontierLabs/coast-sim/issues/91)) ([bfd05df](https://github.com/CosmicFrontierLabs/coast-sim/commit/bfd05df740c23cbc621f8f4c76212914485ec26e))
* fault management refactor ([#99](https://github.com/CosmicFrontierLabs/coast-sim/issues/99)) ([e2a3139](https://github.com/CosmicFrontierLabs/coast-sim/commit/e2a3139e7a176e575f2c4c8ad9dbc91777affd2e))
* improve support for spacecraft roll angles ([#94](https://github.com/CosmicFrontierLabs/coast-sim/issues/94)) ([16c041f](https://github.com/CosmicFrontierLabs/coast-sim/commit/16c041f6444c91b3d06e6ff9c1166cc45fdfa357))
* redo how solar panels are defined ([#95](https://github.com/CosmicFrontierLabs/coast-sim/issues/95)) ([67d1aba](https://github.com/CosmicFrontierLabs/coast-sim/commit/67d1abafbbfa290b8bcb011f86143d9ada09c0df))
* rename `Payload`'s `payload` attribute to `instruments` ([#96](https://github.com/CosmicFrontierLabs/coast-sim/issues/96)) ([0d9dfc1](https://github.com/CosmicFrontierLabs/coast-sim/commit/0d9dfc11d4c74118bed247e80f8c2dc4ede9647e))
* **targets:** Add slew distance weight to target selection scoring ([#70](https://github.com/CosmicFrontierLabs/coast-sim/issues/70)) ([d74803f](https://github.com/CosmicFrontierLabs/coast-sim/commit/d74803faca89edabebe4a8639bcae6c558ec7ba9))


### Bug Fixes

* Add pass-aware scheduling to prevent observation/pass conflicts ([#82](https://github.com/CosmicFrontierLabs/coast-sim/issues/82)) ([5c49ee6](https://github.com/CosmicFrontierLabs/coast-sim/commit/5c49ee6927d739f4881fe87aee582cb53e0500d0))
* Address three misc bugs missed in previous updates ([#83](https://github.com/CosmicFrontierLabs/coast-sim/issues/83)) ([b1865f8](https://github.com/CosmicFrontierLabs/coast-sim/commit/b1865f8cb51e5409e0065a1051b95a0efe2a9b2a))
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
