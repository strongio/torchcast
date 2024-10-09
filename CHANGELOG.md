# CHANGELOG


## v0.4.1 (2024-10-09)

### Continuous Integration

* ci: Update actions/checkout version ([`ed64632`](https://github.com/strongio/torchcast/commit/ed646329cfc2665b2c1732b2c05e7ef30b1f80f6))

* ci: Clone repo using PAT ([`d0adaca`](https://github.com/strongio/torchcast/commit/d0adacac743986e97317ea499b72aee7e6724fc0))

* ci: Enable repo push ([`f565d2a`](https://github.com/strongio/torchcast/commit/f565d2ac262f7096b304b8aac482303492c37895))

* ci: Use SSH Key ([`469d531`](https://github.com/strongio/torchcast/commit/469d53114417314ee28d5fa655b67a6b3310d7e5))

* ci: Fix docs job permissions ([`e6e2e34`](https://github.com/strongio/torchcast/commit/e6e2e346d68725b2ff7eaec57f859079221901cb))

* ci: Pick python version form pyproject.toml ([`2a9eef7`](https://github.com/strongio/torchcast/commit/2a9eef7ca5f609a7fb14197286f72bc6ff095bff))

* ci: Setup auto-release ([`9df4f26`](https://github.com/strongio/torchcast/commit/9df4f2642fe74e6016c2b2a3980ca0eba3c77403))

### Documentation

* docs: Fix examples ([`6f5a2dc`](https://github.com/strongio/torchcast/commit/6f5a2dc5cb8fea4895f44bdba67c60f27e09a84b))

* docs: AirQuality datasets [skip ci] ([`c675f04`](https://github.com/strongio/torchcast/commit/c675f04bb244a73155fbd98c110590bd735808bc))

* docs: Self-hosted docs and fixtures ([`baca184`](https://github.com/strongio/torchcast/commit/baca184beeb53ff44065b6753b220029c5467f9b))

### Fixes

* fix: AQ Dataset ([`9b6e23e`](https://github.com/strongio/torchcast/commit/9b6e23e0ac1a0a7511f490f5b92d3b5e5c69fb59))

### Refactoring

* refactor: Switch to pyproject.toml ([`6de2f27`](https://github.com/strongio/torchcast/commit/6de2f279d82d6fb78e1464aecabdd642969e86e0))


## v0.4.0 (2024-10-03)

### Unknown

* Bump version ([`4563128`](https://github.com/strongio/torchcast/commit/45631283fd31b0d1219595b7c0d698c2e65ef66c))

* Merge pull request #24 from strongio/develop

Develop ([`148af61`](https://github.com/strongio/torchcast/commit/148af6184fe74c795dffae5c2bfea48750740bb4))

* Update data.py ([`e3b3d42`](https://github.com/strongio/torchcast/commit/e3b3d4287731dc232797ac8a1a9a0f55baaba479))

* don't validate args ([`a92ea56`](https://github.com/strongio/torchcast/commit/a92ea5640282161907afb99217483d2fc491c46c))

* improve docstring ([`450f1c2`](https://github.com/strongio/torchcast/commit/450f1c2f7cc2359f1fc0f1eb0fc634c7b8d7a4af))

* cleanup ([`6b05c1f`](https://github.com/strongio/torchcast/commit/6b05c1f419e4303069dc9301f6fe04efc3780d52))

* fix bug in _prepare_initial_state with offsets; fix num_sims>1 ([`cae2879`](https://github.com/strongio/torchcast/commit/cae28795095110da56f081b3e2ec4fd942c546d1))

* bugfixes in _standardize_times ([`4f43a13`](https://github.com/strongio/torchcast/commit/4f43a136424199a840c7e1dd4b7a355d6f9fe146))

* fix tests ([`649611c`](https://github.com/strongio/torchcast/commit/649611cdbd914b87db6887798aa7798d7697e3d8))

* remove _scale_by_measure_var ([`2e342e4`](https://github.com/strongio/torchcast/commit/2e342e41989f2632525c3f22152392666cfd0f45))

* add more metadata to predictions ([`c2a0e05`](https://github.com/strongio/torchcast/commit/c2a0e056a538712609ccaa9cdeee6009fde9baa9))

* remove experimental outlier rejection (for now) ([`07c5ac8`](https://github.com/strongio/torchcast/commit/07c5ac842e7d76f65d9b5c306f73997720d61692))

* clean up the 'get time' helpers; refactor simulate to avoid duplication ([`d79a80f`](https://github.com/strongio/torchcast/commit/d79a80f4acd7b6dc1e45a9c0e2f13891ba720fda))

* improve backwards-compatibility ([`e36e566`](https://github.com/strongio/torchcast/commit/e36e5662b797ab840d16db31a2fbfad861ae2718))

* Merge pull request #23 from strongio/feature/ekf

Feature/ekf ([`bf957dd`](https://github.com/strongio/torchcast/commit/bf957ddc6596849040c428beb1dde027dbd5194b))

* fix tests ([`b0a0931`](https://github.com/strongio/torchcast/commit/b0a09314c09b69ee94c272f68a1d1f6af422efea))

* add EKFPredictions ([`4d8c4e1`](https://github.com/strongio/torchcast/commit/4d8c4e14b5bb0ec33d84d99b67d0a47a24faee13))

* fix conf2bounds call ([`049108c`](https://github.com/strongio/torchcast/commit/049108c3e2caca8318e7f4c496c02c87cfd5fd89))

* fix import ([`e0aebd9`](https://github.com/strongio/torchcast/commit/e0aebd9baf739a0ddde6a72d1f8970c864705e6f))

* remove unneeded abstractions ([`acae72b`](https://github.com/strongio/torchcast/commit/acae72beffd3fb74fe6e0afba9182b32ebaa71a4))

* ekf ([`b40460a`](https://github.com/strongio/torchcast/commit/b40460a1a2c6c616184b777ff25e96e27a91b002))

* fix imports ([`8e6efb5`](https://github.com/strongio/torchcast/commit/8e6efb51cfcac409f058b4496a6dd1481d414312))

* bernoulli filter ([`43ef646`](https://github.com/strongio/torchcast/commit/43ef6461e4b231dd4172063e8d9c1f3fa8502918))

* fix merge ([`8694115`](https://github.com/strongio/torchcast/commit/86941159147ae11ed19b38e99865cf12be0bea18))

* Merge branch 'develop' into feature/poisson ([`4f80b5f`](https://github.com/strongio/torchcast/commit/4f80b5f49f62f9bad0f81154267ee6fec8fcd374))

* abstract into EKFStep ([`cee6f1b`](https://github.com/strongio/torchcast/commit/cee6f1bebdad05038f1a4d1a0315bdbb05975067))

* dont use POISSON_SMALL_THRESH for log_prob ([`ad762d8`](https://github.com/strongio/torchcast/commit/ad762d880165bde2b7d777f75d9dbb9f2d8dd4d6))

* Merge branch 'feature/outliers' into feature/poisson ([`9ee27b1`](https://github.com/strongio/torchcast/commit/9ee27b1ce17b11235c2a98586afb16dbdab9b9d3))

* scipy required ([`1c40d7d`](https://github.com/strongio/torchcast/commit/1c40d7d6288082b5f24215bb8b5a40bbf682b8d2))

* use state-cov when prediction is large ([`ff81fdd`](https://github.com/strongio/torchcast/commit/ff81fdd2adc73efcba0b23d1f26b36545682e896))

* keep multi but use deprecation warning ([`c56f9dc`](https://github.com/strongio/torchcast/commit/c56f9dc1c8c1d289b82555159e8a644b2fda049c))

* update docs ([`546b434`](https://github.com/strongio/torchcast/commit/546b4346687c8c8e2f546e0f34a57c8478b53356))

* Update air_quality.py ([`2fbbcaf`](https://github.com/strongio/torchcast/commit/2fbbcaf3119459939e127eed9a77fbd157dc8b96))

* Merge branch 'main' into feature/poisson ([`8fccac3`](https://github.com/strongio/torchcast/commit/8fccac39f799747ec7c29a358da6fa32b7159e4a))

* Add to docs ([`6e7cc1d`](https://github.com/strongio/torchcast/commit/6e7cc1ddd8c9f766cadc7891b50fa330555ab19a))

* Update air_quality.py ([`548f003`](https://github.com/strongio/torchcast/commit/548f0035f4b65402497e05ad5b57537fac656d6b))

* add warning ([`0ef84e4`](https://github.com/strongio/torchcast/commit/0ef84e46d5806f1973f791340fdde6cd0e65d026))

* Make set_initial_values more flexible ([`e70159c`](https://github.com/strongio/torchcast/commit/e70159c2176198cdce365699925e5e010269b99e))

* Add PoissonFilter ([`b522b1e`](https://github.com/strongio/torchcast/commit/b522b1e045a77eae60eb3eea5b7dc3f876656027))

* move kalman filter ([`ddfbb79`](https://github.com/strongio/torchcast/commit/ddfbb79fa91842d4449bbe6c41a1bc7ccb08b874))

* add _get_quantiles ([`b5425e5`](https://github.com/strongio/torchcast/commit/b5425e5e8570671ed4c7c87156e2c510469682f4))

* more helpful error message ([`63a2978`](https://github.com/strongio/torchcast/commit/63a29780309fd4555ba6caa713c07880872bf68a))

* Update data.py ([`e8108f2`](https://github.com/strongio/torchcast/commit/e8108f299f9dc789a9902ec59d2892f71d40e9c8))

* Update kalman_filter.py ([`5e584e2`](https://github.com/strongio/torchcast/commit/5e584e260ea9c1b08f4a2e7a540a6daf4d1c63d0))

* fix outlier handling in KalmanStep ([`901735c`](https://github.com/strongio/torchcast/commit/901735cb170004ca2dcf13c49dfa978678edbac3))

* allow different thresholds for negative vs positive outliers ([`c1a80c9`](https://github.com/strongio/torchcast/commit/c1a80c99e960ac9a1f2098464dc8037f9f729867))

* add `get_loss` to fit() ([`3468f55`](https://github.com/strongio/torchcast/commit/3468f55b980e8fa0daff84266719c829fefeda24))

* add max_dt_colname to complete_times ([`d234d2e`](https://github.com/strongio/torchcast/commit/d234d2ebe7f32729f90d718317d37e30c3d90c56))


## v0.3.1 (2024-06-17)

### Features

* feat: Move pipeline to Github Actions ([`cf583f8`](https://github.com/strongio/torchcast/commit/cf583f8a8c9e1673b80d16b6ddc5372fe32875db))

### Fixes

* fix: remove test for Python 3.6 ([`6cfa727`](https://github.com/strongio/torchcast/commit/6cfa7274506d09a460cde050469ef6521b637b69))

### Unknown

* Merge pull request #22 from strongio/develop

Develop ([`4cb937b`](https://github.com/strongio/torchcast/commit/4cb937bce130707513d89650121a1f29ab1630ca))

* fix bug in Predictions._subset_to_times ([`f4629ba`](https://github.com/strongio/torchcast/commit/f4629ba002898836e6c5021a995f0643d1b491be))

* Merge branch 'main' into develop ([`1fb4bef`](https://github.com/strongio/torchcast/commit/1fb4bef43074645fe0d5dd08b94a69157cc6a62e))

* enable pip cache ([`976931f`](https://github.com/strongio/torchcast/commit/976931f5f79bd07d9b043070001e49080bdedb65))

* cached_property is available in functools for Python 3.8+ (#21) ([`8c1d53d`](https://github.com/strongio/torchcast/commit/8c1d53d589a0439e7bab2b053c160cba886e5aba))

* require python >= 3.8 ([`82ac0f6`](https://github.com/strongio/torchcast/commit/82ac0f686f569fee2a8f7a0c9edbfe9d2b95a263))

* Only run for push to main branch ([`27cee0e`](https://github.com/strongio/torchcast/commit/27cee0e4cef849dbcd860289a86ab99b5c2e5e52))

* requires 3.8 ([`4491152`](https://github.com/strongio/torchcast/commit/44911522eeda6f32c56bec349f45bee79f370e46))

* Merge remote-tracking branch 'refs/remotes/origin/main' into develop ([`d2a1c81`](https://github.com/strongio/torchcast/commit/d2a1c81dd300a0da3a8a1c124c25b0630e485af8))

* remove python 3.7 support ([`3256cb1`](https://github.com/strongio/torchcast/commit/3256cb18cd1ff0c5b60126808f77ed790da48faf))

* Merge pull request #16 from strongio/develop

Develop ([`74ffd26`](https://github.com/strongio/torchcast/commit/74ffd26a15eb83c4d334d1dccb7beaf9fd7e1ae2))

* Merge pull request #15 from strongio/develop

Develop ([`59bd742`](https://github.com/strongio/torchcast/commit/59bd74288c24cb52d721779e2259fdc11411ac38))

* fix type-hint ([`b676e2c`](https://github.com/strongio/torchcast/commit/b676e2c6005a4f7083548f9222f5e6e591edc9ea))

* fix global_max ([`634a681`](https://github.com/strongio/torchcast/commit/634a681116a93d1bf61563c0b72a671892d3127f))

* provide bad group indices ([`67c2c72`](https://github.com/strongio/torchcast/commit/67c2c72ba6e6a6c8306fafcf19658f6210b06643))

* fix type-hint ([`fd87973`](https://github.com/strongio/torchcast/commit/fd87973bbd8425546509ae459ace1f636b951c03))

* support passing datetime for global max ([`c1cf45a`](https://github.com/strongio/torchcast/commit/c1cf45adb98fbc0adb0c9bbd49982da6cec6d389))

* Merge pull request #14 from strongio/feature/outliers

Outliers ([`644b442`](https://github.com/strongio/torchcast/commit/644b44280e139945f68427ab537c9ed4e5501902))

* update docstring ([`6df683b`](https://github.com/strongio/torchcast/commit/6df683bacbdd389e968bec381ee64e070cafde51))

* rather than dropping outliers, under-weight them

better for lbfgs optimizer ([`e28baf3`](https://github.com/strongio/torchcast/commit/e28baf310dd16d26a33099b0d5c8fcacd46c5c46))

* add experimental note ([`db2f17d`](https://github.com/strongio/torchcast/commit/db2f17d6bde2f8eaca2a284924aad6d777d07085))

* fix merge ([`db63eff`](https://github.com/strongio/torchcast/commit/db63eff3ec091db129fd93d262374e7852fc29a3))

* Merge branch 'develop' into feature/outliers ([`efbf823`](https://github.com/strongio/torchcast/commit/efbf8235508a0030b877c937ef582999dd6e5c0e))

* docstring ([`977ea81`](https://github.com/strongio/torchcast/commit/977ea81a41c5b40723e2098862a97a819cfabf33))

* if the model used an outlier threshold, mask out outliers in the log-prob as well ([`9597f96`](https://github.com/strongio/torchcast/commit/9597f96eb3b0c42457fd42a8e32ad071c567d25f))

* move to _get_update_mask ([`7c1b3c4`](https://github.com/strongio/torchcast/commit/7c1b3c43283279ab5ea149d64fbc127c3f144913))

* outlier-rejection ([`7f511d5`](https://github.com/strongio/torchcast/commit/7f511d5895a29a0fddb67e7007672b09dfa1f79e))

* add ffill option in TimeSeriesDataset.from_dataframe ([`425c648`](https://github.com/strongio/torchcast/commit/425c648e64a8f826d2df45352cc78051b406bdd5))

* add an assert to help with bad measure_cov specs ([`80036cb`](https://github.com/strongio/torchcast/commit/80036cbb7bc05889a6815569d453f3f21bd81017))

* fix bug in predict_variance...

...where we require >0 preds but also exp them ([`496be3e`](https://github.com/strongio/torchcast/commit/496be3e049dcc5d4557ef836fd9ced1c22936c28))

* update complete_times

- global_max
- fix dt_unit == 'W' ([`01fde25`](https://github.com/strongio/torchcast/commit/01fde252a38417fdfa1c8f49592def15496d1734))

* complete_times takes group_colnames ([`338e84d`](https://github.com/strongio/torchcast/commit/338e84d0f4c180592b944dc1694fb3635432bf2c))

* newer version of torch ([`a3ba467`](https://github.com/strongio/torchcast/commit/a3ba467250db9412ffdde64f738ca53ea09c8537))

* revert K ([`c23a7e2`](https://github.com/strongio/torchcast/commit/c23a7e2e6a0675d9cd2072f2ee7b7cd377f9748a))

* Merge pull request #7 from strongio/predictors-example

Predictors example ([`6ef226f`](https://github.com/strongio/torchcast/commit/6ef226f2c10288d21cb9afb2d4916678e1d8b08c))

* Reduce K in seasonality as simpler way of speeding up examples ([`8d80a42`](https://github.com/strongio/torchcast/commit/8d80a42e6947f253a1caf139f041eb97fe832a3a))

* Add a section on external predictors ([`a368c91`](https://github.com/strongio/torchcast/commit/a368c910ae22703b6418518c1b6ebe63ad77814f))

* Add smaller dataset for RTD

rtd debugging ([`b6b62b7`](https://github.com/strongio/torchcast/commit/b6b62b7cf76752258b245d62975a012c62b13d9c))

* Subset to the example group in zoomed in plot ([`f52ba3c`](https://github.com/strongio/torchcast/commit/f52ba3c83752dce899e167862319777da323723a))

* fix var_predict_multi removal ([`1415a51`](https://github.com/strongio/torchcast/commit/1415a51155c99dc56593f4116185d22bd1072de6))

* Remove var_predict_multi ([`4dc6d0e`](https://github.com/strongio/torchcast/commit/4dc6d0ea2a0972d686b4179018df4afcc50ef93b))

* Update setup.py ([`4e3f899`](https://github.com/strongio/torchcast/commit/4e3f899a2e82ef714e0e65b8535c97369813d106))

* Merge pull request #5 from strongio/feature/unbind2

Use unbind; improve electricity example ([`bda7d4a`](https://github.com/strongio/torchcast/commit/bda7d4a931a5b80c05854caf9c8497c03cb407a6))

* Doc fixes ([`04b061d`](https://github.com/strongio/torchcast/commit/04b061d44edd0435b9abd83b61f9c3236f46cce2))

* Fix label in circlci; bump version ([`db06415`](https://github.com/strongio/torchcast/commit/db0641503d436c0cb7547016437469af20776b68))

* Revert "Update rtd.txt"

This reverts commit 390c0ea9fabcbc81e1e022574ad7a11176fe34ef. ([`b34526a`](https://github.com/strongio/torchcast/commit/b34526a5ea06e5b0f7aae6b65d962a1e0cf3a042))

* Update rtd.txt ([`390c0ea`](https://github.com/strongio/torchcast/commit/390c0ea9fabcbc81e1e022574ad7a11176fe34ef))

* Delete sales.py ([`4c24e30`](https://github.com/strongio/torchcast/commit/4c24e308abeeaa914d328a0f24360e8693cafd14))

* switch to jupytext percent-script ([`5100f25`](https://github.com/strongio/torchcast/commit/5100f25ef51911cf38495557ffa87255f37c8403))

* Update requirements ([`8a96439`](https://github.com/strongio/torchcast/commit/8a964398df7bea6d1faeb6f85128a9aeb3fcd61f))

* avoid torch deprecation warnings ([`b800fe6`](https://github.com/strongio/torchcast/commit/b800fe64bb13a6f2c0481da86277701833e0e130))

* Fix tests ([`c7c8de8`](https://github.com/strongio/torchcast/commit/c7c8de8bb07b5ae493c98a74ecfcbaac9ddab80d))

* add pickled models; notebook cleanup ([`b3c3a70`](https://github.com/strongio/torchcast/commit/b3c3a7041ed05f2a304d22e84586e88e4eb48ec0))

* handle X_colnames is none properly when measure_colnames passed ([`c6ba860`](https://github.com/strongio/torchcast/commit/c6ba86084d5a627c94e6b6c72b3dc9c216c133dd))

* Update electricity.py ([`ea953cb`](https://github.com/strongio/torchcast/commit/ea953cba3533299fdd4a9331bcfa73829a3ca599))

* tweak fit ([`4e33074`](https://github.com/strongio/torchcast/commit/4e330747e4ef3f1bb08f8dc6eba144123a393782))

* cleanup ([`c73cb9d`](https://github.com/strongio/torchcast/commit/c73cb9d49b92c3165c0e12c2e73515b43df530c5))

* Update electricity.py ([`7ff2a5f`](https://github.com/strongio/torchcast/commit/7ff2a5feb4a9059d1448e2725a2707d4e966bf6c))

* Make sure add_season_features handles non-default index ([`a7f34d2`](https://github.com/strongio/torchcast/commit/a7f34d2a2c07cd686512bc5ed1f66828c978bb27))

* fix TimeSeriesDataset.get_groups ([`fae3dcb`](https://github.com/strongio/torchcast/commit/fae3dcb406fe1b89a733a7260c368aa3b2cb56bb))

* save kf attempt ([`0dd1f4f`](https://github.com/strongio/torchcast/commit/0dd1f4fe9e75b2826e79676e29960bab3df4b760))

* different approach

This reverts commit ea2ec99249ccf14582dbde141360a1629ecf272f. ([`01d16e9`](https://github.com/strongio/torchcast/commit/01d16e9945525b6c452a97aa86ac73638774f667))

* can't convert cuda:0 device type tensor to numpy ([`ea2ec99`](https://github.com/strongio/torchcast/commit/ea2ec99249ccf14582dbde141360a1629ecf272f))

* fix nan handling in LinearModel._l2_solve ([`93daf84`](https://github.com/strongio/torchcast/commit/93daf8420373447c2b6f12d8dbf1ae3314f8292e))

* fix singleton handling in validate_gt_shape ([`d2094bb`](https://github.com/strongio/torchcast/commit/d2094bb22e49eda3d98b8b31fc048c601311db8d))

* Update electricity.py ([`7f780da`](https://github.com/strongio/torchcast/commit/7f780da716e1f1b95a0e9377c6d4c6fb658a3206))

* if input has singleton trailing dim, expand ([`d59b734`](https://github.com/strongio/torchcast/commit/d59b734962723e552e7a9dbca63d0341ad8478f9))

* Update electricity.py ([`4f7a447`](https://github.com/strongio/torchcast/commit/4f7a447ed4e3cca9acee33daf22c65bc8b818afe))

* wip updated elec example ([`f4d336d`](https://github.com/strongio/torchcast/commit/f4d336d303c73035941a6cf4238f6903ccfb3b22))

* Update setup.py ([`4378ce0`](https://github.com/strongio/torchcast/commit/4378ce053b1ddc792edf6bfc3dd3b78b9d008bff))

* another dtype [ci skip] ([`946ae5a`](https://github.com/strongio/torchcast/commit/946ae5a49f8ac8d76c7e185c4f75866c3fbf180f))

* match dtype/device ([`5ce11b1`](https://github.com/strongio/torchcast/commit/5ce11b1f2de2f92b04606de6fa0fd75c7736a2fc))

* 36 compat ([`4e76aac`](https://github.com/strongio/torchcast/commit/4e76aac9010d0117395c634099b811181e610e8d))

* allow passing just y_colnames ([`73a1222`](https://github.com/strongio/torchcast/commit/73a1222ef5c8570378f0a5dd2124054c2a36dc47))

* Add LinearModel.solve_and_predict ([`3f58569`](https://github.com/strongio/torchcast/commit/3f58569139719ef00d209a8bd41d64ea41456e75))

* ExpSmoothStep allows passing None instead of a tens of zeros ([`c74a94d`](https://github.com/strongio/torchcast/commit/c74a94d0e50f3cc71eef50dbb1385d9709f89fa7))

* Pass lists instead of pre-stacking; add todos ([`f2e8c58`](https://github.com/strongio/torchcast/commit/f2e8c58a8a5906d6561b67df7b95eb1d0952729c))

* wip rossman example ([`95ec117`](https://github.com/strongio/torchcast/commit/95ec11738b084a6eed543a15b6468ef991bb09d6))

* wip update to electricity ex ([`e718355`](https://github.com/strongio/torchcast/commit/e718355755f5e32795697f51d56eaca5b47c7177))

* permit collapsing across state elements ([`8d926b7`](https://github.com/strongio/torchcast/commit/8d926b7eb830b2df94eadd1d72f7fe8063f35623))

* Fix bug in with_new_start_times(quiet=True) ([`40d4618`](https://github.com/strongio/torchcast/commit/40d4618d99b549d9c27615f144962967d76f61fb))

* fix bug in forward when input is exhausted ([`bf03f23`](https://github.com/strongio/torchcast/commit/bf03f235bbfd891d9f7faf473d1702ceab598787))

* Update smoothing_matrix.py ([`754c370`](https://github.com/strongio/torchcast/commit/754c37011e4392f3a9e6bec1c0b6570b2ad6901b))

* fix test_fourier_season ([`77964b4`](https://github.com/strongio/torchcast/commit/77964b4e2f50d994ade3c332269781b1b5a1143e))

* Update test_training.py ([`45450b9`](https://github.com/strongio/torchcast/commit/45450b9726efdbe8f9dc375005e7a2f2c60e07b4))

* Update base.py ([`eab1747`](https://github.com/strongio/torchcast/commit/eab174730d121e6c8a42abac34aa3d1285e1ae6e))

* redefine initial state; fix torchscript issue ([`5103404`](https://github.com/strongio/torchcast/commit/510340461e136742ff101c132e5a3e1541d0c1a9))

* don't fail on if missing optional dependency ([`a4d4bfe`](https://github.com/strongio/torchcast/commit/a4d4bfe3895018c7b073cc32a92c1c2d73c270e9))

* wip rework of electricity example ([`da85703`](https://github.com/strongio/torchcast/commit/da857035e2d2eb5c36c9fb669bb41fc1d24b27a1))

* Fix ExpSmoother ([`781493c`](https://github.com/strongio/torchcast/commit/781493c5a93e10fe5e15b6ce309219e49be9f863))

* Fixes/updates ([`2d09dc5`](https://github.com/strongio/torchcast/commit/2d09dc5ccbae96aca810f788334a369574d379ae))

* WIP refactor to simplify design-mat creation w/unbind ([`d98b47d`](https://github.com/strongio/torchcast/commit/d98b47d1ac38c61653093efd887cc672f228917d))

* Improvements to covariance ([`8880c7a`](https://github.com/strongio/torchcast/commit/8880c7a14cf77655f500f452fcc29cf2825c4422))

* Bugfixes related to dt_unit ([`6f183bb`](https://github.com/strongio/torchcast/commit/6f183bb41db45e187dfbe4277fe6695a3a392132))

* Predictions._subset_to_times handles dt_unit is None ([`8cb6b79`](https://github.com/strongio/torchcast/commit/8cb6b7920cb7ff02cb44a7bab8ab482bb633be0c))

* Split covariance into module ([`e7ecd88`](https://github.com/strongio/torchcast/commit/e7ecd88cacdb295764e745b9103c9ee3c53cc0f8))

* Don't try to set iniitial values if not in state-dict ([`0033758`](https://github.com/strongio/torchcast/commit/00337583478a6653b361a6ffa338ceafdf007a6b))

* fix SmoothingMatrix(method='full') ([`ec20e28`](https://github.com/strongio/torchcast/commit/ec20e2812828fda7a9106065672ad678fb49f3b8))

* Improve support for low-rank K in exp-smooth ([`f47fb0e`](https://github.com/strongio/torchcast/commit/f47fb0ea7f8de78c4034c228363a96dc283c6d75))

* support choosing low-rank for covariance-matrix ([`84529db`](https://github.com/strongio/torchcast/commit/84529db4871282891aa9de4d45dc15b2d165ee20))

* support low-rank K in exp-smooth; rename smoothing mat ([`b5f8ddd`](https://github.com/strongio/torchcast/commit/b5f8dddf1e50bdbdbf03ae69681ca1260df6c8ee))

* Fix handling of datetime.datetime ([`a75c55d`](https://github.com/strongio/torchcast/commit/a75c55d5d3a47fa8e0e0f83d3bd108e3679be7ed))

* Merge pull request #4 from strongio/feature/expsmooth

Add exponential smoothing ([`bbef877`](https://github.com/strongio/torchcast/commit/bbef87701551dc2a262959304f73960309805eed))

* Doc tweaks ([`ab51eb7`](https://github.com/strongio/torchcast/commit/ab51eb7deffa85e8dcf766b2927517b77ac9fac6))

* Make build_design_mats private; rename to ExpSmoother ([`ae54b9a`](https://github.com/strongio/torchcast/commit/ae54b9a844dc1fb8d3b828cfe497f91fb47b0ab4))

* Update electricity.py ([`8901d9a`](https://github.com/strongio/torchcast/commit/8901d9ab3f63fb8d1bb638d11b73717124800300))

* Merge branch 'develop' into feature/expsmooth ([`08ee42c`](https://github.com/strongio/torchcast/commit/08ee42c01787930e27337e22d72e4ba1daa28b0c))

* Merge branch 'main' into develop ([`9f2ba74`](https://github.com/strongio/torchcast/commit/9f2ba74fa2921e70056599dd69d0edac99759af1))

* Update exp_smooth.py ([`0bac251`](https://github.com/strongio/torchcast/commit/0bac25150e02c83e052de086e6a5d7ad306ee008))

* Better inits for K ([`ec05852`](https://github.com/strongio/torchcast/commit/ec0585238e8a05ab10689867593e108b38e9a154))

* Fix simulate ([`acaa4f9`](https://github.com/strongio/torchcast/commit/acaa4f9671d1e6fbdfac13a0d1dc17e5ab5be959))

* Working ExpSmooth ([`ecf0c67`](https://github.com/strongio/torchcast/commit/ecf0c670043427ceb99f0d477bab626527ce8f20))

* wip: implement exp-smooth ([`8210da6`](https://github.com/strongio/torchcast/commit/8210da6ee85f13f07cdbd7b237b9abc6ad07be3b))

* Fix bugs from renaming ([`666acc4`](https://github.com/strongio/torchcast/commit/666acc4e9c39fe9114cb7c07b417dc26492e013b))

* Merge branch 'main' into feature/expsmooth ([`6b9daff`](https://github.com/strongio/torchcast/commit/6b9daff1e0097ebd863f5ccc7c4814b0227e7932))

* typo ([`ce96f7b`](https://github.com/strongio/torchcast/commit/ce96f7b22e16683f0265080b8a5778b4f419ef54))

* GaussianStep -> KalmanStep ([`2c89f5f`](https://github.com/strongio/torchcast/commit/2c89f5fdb72eb51e68f71a399283d7e59974cdeb))

* Avoid referring to proc-variance in Processes ([`a42e61c`](https://github.com/strongio/torchcast/commit/a42e61c6fe6fbd6f136c0ae16a1df8b37cd3a851))

* Move update validation to base ss-step cls ([`cf9b7be`](https://github.com/strongio/torchcast/commit/cf9b7beb927b47f35e4866d44c09d6cdcf4abe56))

* WIP Update inheritance

base class only assumes a measure cov ([`c715cbd`](https://github.com/strongio/torchcast/commit/c715cbde9c48fc79cbd79b4bcf347f77e6d6e642))

* WIP exp-smooth algorithm ([`d0b2a5e`](https://github.com/strongio/torchcast/commit/d0b2a5ef3ebe6cbaf7ebf54acf56be1dd5045ed8))

* Update setup.py ([`a53f56b`](https://github.com/strongio/torchcast/commit/a53f56bbc4bcec57822c499933035d2263331a8d))

* make state-space more extensible ([`9b183ab`](https://github.com/strongio/torchcast/commit/9b183ab1bf1805917f6e9d1590adbfd58544c69b))

* Fix group_colname ([`d09e5b8`](https://github.com/strongio/torchcast/commit/d09e5b85420c4d4068f1a8890957d86ccd8964f6))

* move electricity models to dir ([`b53bf58`](https://github.com/strongio/torchcast/commit/b53bf58ccf63680d1bf2d721dd9b689c8c519559))

* Update electricity.py ([`bfd5410`](https://github.com/strongio/torchcast/commit/bfd5410b3a956f12fc7f835b2d57287bd46043ce))

* Update README.rst ([`03b29b3`](https://github.com/strongio/torchcast/commit/03b29b3e85ad6d15bf6ba7197ac822de69e83b34))

* Typos; add strong link ([`a347db5`](https://github.com/strongio/torchcast/commit/a347db576bc0b35a87d654366311059c263fdad1))

* Merge pull request #1 from strongio/feature/statespace-base

make StateSpaceModel the base cls ([`1c036df`](https://github.com/strongio/torchcast/commit/1c036df546b9461446006c161401119e6ff98220))

* Update docs ([`0f050bc`](https://github.com/strongio/torchcast/commit/0f050bc450b827bfe21c76be70fe809d6895daa9))

* make StateSpaceModel the base cls ([`a9778bd`](https://github.com/strongio/torchcast/commit/a9778bdee9b3a34da340591cf9a90db0ccfc52d8))

* typo ([`81f703f`](https://github.com/strongio/torchcast/commit/81f703f5988ed7d8abffcdeccec291e198c1393e))

* Add setup.cfg ([`3022069`](https://github.com/strongio/torchcast/commit/3022069e201e5905a1955e75ed8f4f77674bb12a))

* Update config.yml ([`7e5f44b`](https://github.com/strongio/torchcast/commit/7e5f44b63cb83fa8ad90b0841407ef87dcd70a07))

* Update config.yml ([`10c400b`](https://github.com/strongio/torchcast/commit/10c400b1db299b276f07cab7564037427289a201))

* Create config.yml ([`03d3956`](https://github.com/strongio/torchcast/commit/03d3956fad396c2ec41b00a1bf95e4cd380469cc))

* Add image to readme ([`83312ef`](https://github.com/strongio/torchcast/commit/83312ef4b87ff6dddef7b842e845fb049cf0354f))

* Fix doc links in readme ([`1946b8a`](https://github.com/strongio/torchcast/commit/1946b8a356220064cf9d9b1b0a27e8e13a471015))

* Reduce batch size ([`ed82c95`](https://github.com/strongio/torchcast/commit/ed82c953d05087b4444f8ba2ac405b0b43db024c))

* Update doc requirements ([`2458895`](https://github.com/strongio/torchcast/commit/24588957b5916cb803e6fa59ca5ccfc2f13a8a15))

* Update setup.py ([`b9fde1a`](https://github.com/strongio/torchcast/commit/b9fde1a53b1f04957bfe5406a6916b019e9b4969))

* Update conf.py ([`ca38e9e`](https://github.com/strongio/torchcast/commit/ca38e9e629300f1b7e215db3ccd726e82099fec3))

* Add ipykernel ([`cb268e9`](https://github.com/strongio/torchcast/commit/cb268e9e525ee475ce38ab2732e6b31273c5b7c7))

* Use rtd.txt for build ([`eefa1d1`](https://github.com/strongio/torchcast/commit/eefa1d16c2c464c2cd584d832480be7aefa76bd5))

* Remove duplicate files ([`97b5ff4`](https://github.com/strongio/torchcast/commit/97b5ff40be61f2a040c4f6787e14e0a2b795aba2))

* Remove kernelspecs ([`6fa0e1d`](https://github.com/strongio/torchcast/commit/6fa0e1d6c9dac262e36b963ac3d417ce7bd014de))

* Update requirements.txt ([`ab4cef3`](https://github.com/strongio/torchcast/commit/ab4cef3e58a389566987bb5d2793af490815dd55))

* Merge branch 'main' of https://github.com/strongio/torchcast into main ([`d3c6553`](https://github.com/strongio/torchcast/commit/d3c65534791debf39946641fc9c38e952f432a6f))

* Update README.rst

Fix typo ([`153e580`](https://github.com/strongio/torchcast/commit/153e580af5b2e9d29e9690081c7ff88ed5b40ba7))

* Add files for building docs ([`4cc6dcc`](https://github.com/strongio/torchcast/commit/4cc6dccec380677d8a751570e90c61e6dd210045))

* Docs ([`393489c`](https://github.com/strongio/torchcast/commit/393489ce62e52b3d36df9848f7a44709e687fa0f))

* Initial commit ([`39f178d`](https://github.com/strongio/torchcast/commit/39f178d712794544d50fca6c1afcfe2610272a68))
