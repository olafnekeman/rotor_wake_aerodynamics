C
C   PPLOT array limits
C
C     NAX    number of points in one polar
C     NPX    number of polars
C     NFX    number of points in one reference polar
C     NDX    number of reference polars
C     ISX    number of airfoil sides
C
C
      INCLUDE 'PINDEX.INC'
C
      PARAMETER (NAX=500, NPX=20, NFX=128, NDX=20, ISX=10)
      CHARACTER*80 FNPOL, FNREF, FNAME
      CHARACTER*32 NAME, LABREF
      CHARACTER*80 TITLE
      CHARACTER*16 CODE, CCLEN
      LOGICAL LGRID,LPLOT, LCDW, LLIST, LEGND, LCLEN, LAECEN, AUTO
      REAL MACH
C
      COMMON/ALLI/
     &  IDEV, IDEVRP, IPSLU, ICOL0, NCOLOR,
     &  NA(NPX), NPOL, ICOL(NPX), ILIN(NPX),
     &  NF(4,NDX), NDAT, IFCOL(NDX), IFSYM(NDX), NBL(NPX),
     &  IRETYP(NPX),IMATYP(NPX), NCLEN,
     &  LGRID,LPLOT,LCDW,LLIST,LEGND,LCLEN,LAECEN, AUTO
C
      COMMON/ALLR/
     &  PLOTAR, CH, CH2,
     &  SIZE,SCRNFR,
     &  CPOLPLF(3,IPTOT), VPOLPLF(3,2), 
     &  XCD,XAL,XOC,
     &  MACH(NPX), REYN(NPX), ACRIT(NPX),XTRIP(ISX,NPX),
     &  CPOL(NAX,IPTOT,NPX),
     &  CPOLSD(NAX,ISX,JPTOT,NPX),
     &  CDLMOD(5,NPX),VPPARS(6,NPX),
     &  XYREF(NFX,2,4,NDX),
     &  VERSION
C
      COMMON/ALLC/
     &  FNPOL(NPX), NAME(NPX),
     &  FNREF(NDX), LABREF(NDX), TITLE, CODE, CCLEN, FNAME
C









