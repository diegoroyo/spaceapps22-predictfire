# Wildfire prediction with NASA's open data

Team: Sin Rocket no hay Paraíso
* Diego Royo
* David Ubide
* David Morilla
* Lorenzo Cano
* Andrés Fandos
* Pedro Orús

TL;DR

* Cell simulation with Rothermel's "simple" fire model.
* It takes as input several variables about the environment and weather
(e.g. fuel load, depth and moisture, wind speed, terrain slope), and outputs
the rate of spread of the fire (m/s).
* We collected environment and weather information using NASA's open data
(Earthdata Search, GIBS, CMR) for the Moncayo region in Aragón, Spain
* From an initial ignition source, our simulation shows which parts of the map
will be burned, fastest spread zones, safe zones, etc.
* Terrain properties can be changed (e.g. simulate drier terrain caused by
climate change) to see how it affects fire spread, leaving other conditions
unchanged.
