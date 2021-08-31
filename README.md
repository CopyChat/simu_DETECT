# simu_DETECT
simulation for DETECT project

`Project  description`


*this simulation is run @CCuB*


### Simulation forcing data

ERA5 @27km, @3h resolution.


### domain setting:

`downscale ratio = 3, 3, 3, 1km @Reunion`

### first test run setting:

- simulation period: 2021-05-01 to 2021-08-21T21:00:00

- output: 
  - GHI (SWDOWN)
  - DIF (SWDDIF)
  - TEMP (t2m)
- spatial resolution: @1km
- temporal resolution: 10 minutes`

### physical option:

`physics_suite = 'CONUS'`

where 'CONUS' is equivalent to

 - mp_physics         = 8,

 - cu_physics         = 6, (closed for 1 km and 3 km domains)

 - ra_lw_physics      = 4,

 - ra_sw_physics      = 4,

 - bl_pbl_physics     = 2,

 - sf_sfclay_physics  = 2,

 - sf_surface_physics = 2.




