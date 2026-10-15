;========================================================
; This file reads in a traj file produced by the TC tracker
; and then extract the TC wind field
; 1. regrid from ne120 to gx1v6
; 2. find the lat/lon of TC center, define a region surrouding the center
; 3. find the region on the gx1v6 grid and write out
;========================================================
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
load "/glade/u/home/zarzycki/ncl/functions/surfaceWindCorrFactor.ncl"
load "/glade/u/home/zarzycki/ncl/functions/getTrajectoriesGeneric.ncl"
load "/glade/u/home/zarzycki/ncl/functions/mask_tc.ncl"

;========================================================
begin

;sed -i.bak -e '5791,204899d' trajectories.txt.TEST.NE120.1979_2012.001

;thefile="2trajectories.txt.TEST.NE120.0080_0089.005"
ncols = 10                    ; assign number of columns
maxLines = -1
headerDelimStr = "start"
delimStr="tab"
isHeader=False
traj_data = getTrajectoriesGeneric(thefile,ncols,maxLines,headerDelimStr,delimStr,True,isHeader)

lon = ndtooned(traj_data(:,2,:))
lat = ndtooned(traj_data(:,3,:))
wind = ndtooned(traj_data(:,5,:))
pres = ndtooned(traj_data(:,4,:)/100.)
yyyy=  ndtooned(traj_data(:,7,:))
mm= ndtooned(traj_data(:,8,:))
dd= ndtooned(traj_data(:,9,:))
hh= ndtooned(traj_data(:,10,:))


YYYYMMDDHH=1000000*toint(yyyy)+10000*toint(mm)+100*toint(dd)+toint(hh)

f1=addfile(ubot_file,"r")
f2=addfile(vbot_file,"r")
f3=addfile(u10_file,"r")

; every 6hour
; for ax case, the time starts at the 3rd hour
stride = 2
time0=f1->time
ntime = dimsizes(time0)
time = time0(1:ntime-1:stride)
UBOT_SE=f1->UBOT(1:ntime-1:stride,:)
VBOT_SE=f2->VBOT(1:ntime-1:stride,:)
U10_SE = f3->U10(1:ntime-1:stride,:)
;UBOT_SE=f1->UBOT(::stride,:)
;VBOT_SE=f2->VBOT(::stride,:)
;U10_SE = f3->U10(::stride,:)
printVarSummary(U10_SE)

delete(f1)
delete(f2)
delete(f3)


wgt_file="~/tc_tracking/grids/ne120_to_gx1v6_patch_weight.nc"
Opt=True
U10  = ESMF_regrid_with_weights(U10_SE,wgt_file,Opt)
delete(U10_SE)
UBOT  = ESMF_regrid_with_weights(UBOT_SE,wgt_file,Opt)
delete(UBOT_SE)
VBOT  = ESMF_regrid_with_weights(VBOT_SE,wgt_file,Opt)
delete(VBOT_SE)

delete(UBOT@lat2d)
delete(VBOT@lat2d)
delete(U10@lat2d)

delete(UBOT@lon2d)
delete(VBOT@lon2d)
delete(U10@lon2d)

UBOT!1 = "nlat"
UBOT!2 = "nlon"

VBOT!1 = "nlat"
VBOT!2 = "nlon"

U10!1 = "nlat"
U10!2 = "nlon"

printVarSummary(UBOT)


f = addfile("~/tc_tracking/grids/gx1v6_grid.nc","r")
lat2d = f->TLAT
lon2d = f->TLONG

; UBOT@lon2d   = lon2d
; UBOT@lat2d   = lat2d

; VBOT@lon2d   = lon2d
; VBOT@lat2d   = lat2d

; U10@lon2d   = lon2d
; U10@lat2d   = lat2d

printVarSummary(UBOT)
ntime=dimsizes(time)

delete(f)


lat2d_1d = ndtooned(lat2d)
lon2d_1d = ndtooned(lon2d)

dims = dimsizes(lat2d)
printVarSummary(lat2d)


; loop through all the dates to see if the day has TCs
do ii = 0,ntime-1
  print(ii+"")
  ncdate=cd_calendar(time(ii),-3)
  MATCHES=where(YYYYMMDDHH.eq.ncdate,YYYYMMDDHH,YYYYMMDDHH@_FillValue)
  iz = ind(.not.ismissing(MATCHES))
  
  utdate = cd_calendar(time(ii),0)
  yyy = toint(utdate(0,0))
  mmm = toint(utdate(0,1))
  ddd = toint(utdate(0,2))
  dyear = day_of_year(yyy,mmm,ddd)

  if (.not.all(ismissing(iz))) then
    niz = dimsizes(iz)
    MATCHLAT=lat(iz)
    MATCHLON=lon(iz)

 
;begin to extract surrounding wind
; define output ubot and vbot on gx1v6 grid
      UBOT_TC = UBOT(ii,:,:)
      VBOT_TC = VBOT(ii,:,:)
      U10_TC = U10(ii,:,:)
      UBOT_TC = 0.
      VBOT_TC = 0.
      U10_TC = 0.
     

    do jj = 0,niz-1

      blon = MATCHLON(jj) - 10
      tlon = MATCHLON(jj) + 10
      blat = MATCHLAT(jj) - 10
      tlat = MATCHLAT(jj) + 10

; now find the sub-region on the curviliner grid
      ind_lat = ind(lat2d_1d.ge.blat.and.lat2d_1d.lt.tlat)
      lon_sub = lon2d_1d(ind_lat)
      ind_lon = ind(lon_sub.ge.blon.and.lon_sub.lt.tlon)
      final_ind = ind_lat(ind_lon)

      indices  = ind_resolve(final_ind,dims)
      npts = dimsizes(indices(:,0))

    do kk = 0, npts-1
    UBOT_TC(indices(kk,0),indices(kk,1)) = (/UBOT(ii,indices(kk,0),indices(kk,1))/)
    VBOT_TC(indices(kk,0),indices(kk,1)) = (/VBOT(ii,indices(kk,0),indices(kk,1))/)
    U10_TC(indices(kk,0),indices(kk,1)) = (/U10(ii,indices(kk,0),indices(kk,1))/)
    end do


; we want to maintain the maximum wind speed
; replace the maximum wind to the original magnitude
; but in the model, the maximum wiind speed will in fact be underestimated due to regridding. so we may just keep it. 


    delete(ind_lat)
    delete(ind_lon)
    delete(lon_sub)
    delete(final_ind)
    delete(indices)

    end do

  ;   wks = gsn_open_wks("X11","~/homme_"+tostring(ncdate))              ; send graphics to PNG file

  ;   res                     = True         ; plot modifications desired
  ; ;  res@gsnMaximize         = True         ; Maximize size of plot in frame
  ;   res@gsnDraw=False
  ;   res@gsnFrame=False


  ;   res@cnFillOn            = True         ; Turn on contour fill
  ;   res@cnFillPalette       = "cmp_b2r"  ; set color map
  ;   res@cnFillMode          = "RasterFill"   ; Style of fill. You can also
  ;                                          ; use "CellFill" and "RasterFill"
  ;   res@cnLinesOn           = False        ; Turn off contour lines
  ;   res@cnLineLabelsOn      = False        ; Turn off contour line labels

  ;   res@tiMainString        = tostring(ncdate)

  ;   res@cnLevelSelectionMode = "ManualLevels"	; manually set the contour levels with the following 3 resources
  ;   res@cnMinLevelValF  = -20.			; set the minimum contour level
  ;   res@cnMaxLevelValF  = 20.			; set the maximum contour level
  ;   res@cnLevelSpacingF = 1			; set the interval between contours

  ;   map = gsn_csm_contour_map(wks,UBOT_TC(:,:),res)
  ;   map = gsn_csm_contour_map(wks,VBOT_TC(:,:),res)
  ;   dum = new(niz,"graphic")
  ;   pmres = True
  ;   pmres@gsMarkerColor = "black"
  ;   pmres@gsMarkerIndex = 4

  ;   do jj = 0,niz-1
  ;     dum(jj) = gsn_add_polymarker(wks, map, MATCHLON(jj), MATCHLAT(jj), pmres)
  ;   end do
  ;   draw(map)
  ;   frame(wks)

    delete(MATCHLAT)
    delete(MATCHLON)

;    delete(dum)


   fout = addfile("/glade/work/huili7/tc_wind_forcing/tc_files/b.e13.B1850C5CN.ne120_g16.005.UVBOT."+tostring(ncdate)+".nc","c")
  fout->time = time(ii)
  fout->date = ncdate
  fout->dyear = dyear
  fout->UBOT = UBOT_TC
  fout->VBOT = VBOT_TC
  fout->U10 = U10_TC
  print("finish writing")
  delete(UBOT_TC)
  delete(VBOT_TC)
  delete(U10_TC)
  delete(fout)
  end if

    delete(iz)
    delete(MATCHES)

end do



end
