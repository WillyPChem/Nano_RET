/*
  NanoRET.c - a program for simulating quantum resonant energy transfer between a plasmonic nanoparticle
              donor and a small-molecule acceptor.
  
    Copyright (C) 2017  Jonathan J. Foley IV, Matthew Micek, Noor Eldabagh

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Electronic Contact:  foleyj10@wpunj.edu
    Mail Contact:   Prof. Jonathan Foley
                    Department of Chemistry, William Paterson University
                    300 Pompton Road
                    Wayne NJ 07470

*/

#include <fftw3.h>
#include<math.h>
#include<stdio.h>
#include<malloc.h>
#include<complex.h>
#include<string.h>
#define REAL 0
#define IMAG 1
//void Commutator (int dim, double H[dim][dim], double D[dim][dim], double P[dim][dim]);
void RK3(int Nlevel, double time, double *bas, double *E, double *Hint, double *Mu, double *Dis, double complex *D, double dt);
void RK3_MG(int Nlevel, double time, double *bas, double *E, double *Hint, double *Mu, double *Dis, double complex *D, double dt);
void Liouville(int dim, double complex *H, double complex *D, double complex *P);
void AntiCommutator(int dim, double *H, double complex *D, double complex *P);
void PrintComplexMatrix(int dim, double complex *M);
void PrintRealMatrix(int dim, double *M);
void FormDM(int dim, double complex *C, double complex *Ct, double complex *DM);
void L_Diss_MG(int Nlevel, double gamma, double complex *D, double *bas, double complex *P, int ex, int dex);
void L_Diss(int Nlevel, double *gamma, double complex *D, double *bas, double complex *P);
void Fourier (double complex *dm, int n, double dt, double complex *ftout, double *freqvec);
double complex TrMuD(int Nlevel, double *Mu, double complex *D);
double E_Field(double time);
void FillDFTArray(int fftw_iter, double real_val, double imag_val, fftw_complex* ar);
double D_Error(int dim, double complex *D);
// Function prototype for H_interaction
void H_interaction(int dim, double *Hint, double *mu, double dpm, double *R);

// NOTE!!!  You need three global variables for the rates associated with 
// The Lindblad operators - gamma, beta, and alpha should be defined here according
// to their values in the journal of physical chemistry letters paper
//int Nlevel = 3;
//int dim = Nlevel*Nlevel;

double pi = 4*atan(1.0);
double wn_to_au = 4.55633528e-6; 
double mu_au_to_si = 8.47835326e-30; // 1 a.u. = 8.47835326e-30 C m
double E_au_to_si = 5.14220652e11;  // 1 a.u. = 5.14220652e11 V/m
double omega_au = 4.134147e+16;;
double bohr_r_nm = 0.0529177;
double NA = 6.022e-23;
double c = 299792458.;
double RI;
int main() {

  //Nanoparticles Variables here
  //int numTime = 10000000;
  //int zeropad = 10000000;
  int numTime = 100000;
  int zeropad = 100000;

  // COUPLED DONOR VARIABLES
  double *E, *Mu, *Dis, *bas, *Hint;
  double complex *H, *D, *P;

  //double dt = 0.001;
  double dt = 0.01;
  int Nlevel, dim;
  
  // NP levels can be variable in principle
  Nlevel=2;
  dim = Nlevel*Nlevel;
  // COUPLED ACCEPTOR VARIABLES!
  int NlevelMG = 2;
  int dimMG = NlevelMG*NlevelMG;
  double *EMG, *MuMG, *MuZERO, *DisMG, *basMG, *HintMG;
  double complex *HMG, *DMG, *PMG;

  // UNCOUPLED DONOR AND ACCEPTOR DENSITY MATRIX
  double complex *DD, *DA;
  double *Hint_DA, *Hint_AD;

  int dft_dim = numTime+zeropad;
  // FFTW variables here -> inputs to fft
  fftw_complex *dipole;
  fftw_complex *efield;
  fftw_complex *dipoleMG;
  fftw_complex *nps;
  fftw_complex *mgs;
  fftw_complex *efs;

  // FFTW VARIABLES OF UNCOUPLED VARIABLES
  fftw_complex *dipole_donor;
  fftw_complex *dipole_acceptor;
  fftw_complex *alpha_donor;
  fftw_complex *alpha_acceptor;

  // Allocate memory for FFTW arrays
  dipole = (fftw_complex*)malloc(dft_dim*sizeof(fftw_complex));
  efield = (fftw_complex*)malloc(dft_dim*sizeof(fftw_complex));
  dipoleMG = (fftw_complex*)malloc(dft_dim*sizeof(fftw_complex));
  nps = (fftw_complex*)malloc(dft_dim*sizeof(fftw_complex));
  mgs = (fftw_complex*)malloc(dft_dim*sizeof(fftw_complex));
  efs = (fftw_complex*)malloc(dft_dim*sizeof(fftw_complex));

  // Allocate memory for FFTW arrays of uncoupled variables
  dipole_donor    = (fftw_complex*)malloc(dft_dim*sizeof(fftw_complex));
  dipole_acceptor = (fftw_complex*)malloc(dft_dim*sizeof(fftw_complex));
  alpha_donor     = (fftw_complex*)malloc(dft_dim*sizeof(fftw_complex));
  alpha_acceptor  = (fftw_complex*)malloc(dft_dim*sizeof(fftw_complex));

  fftw_plan npp = fftw_plan_dft_1d(dft_dim,
                                      dipole,
                                      nps,
                                      FFTW_BACKWARD,
                                      FFTW_ESTIMATE);

  fftw_plan mgp = fftw_plan_dft_1d(dft_dim,
                                      dipoleMG,
                                      mgs,
                                      FFTW_BACKWARD,
                                      FFTW_ESTIMATE);


  fftw_plan efp = fftw_plan_dft_1d(dft_dim,
                                      efield,
                                      efs,
                                      FFTW_BACKWARD,
                                      FFTW_ESTIMATE);

  fftw_plan donp = fftw_plan_dft_1d(dft_dim,
                                      dipole_donor,
                                      alpha_donor,
                                      FFTW_BACKWARD,
                                      FFTW_ESTIMATE);


  fftw_plan accp = fftw_plan_dft_1d(dft_dim,
                                      dipole_acceptor,
                                      alpha_acceptor,
                                      FFTW_BACKWARD,
                                      FFTW_ESTIMATE);

  // Allocate memory for all other arrays
  // NP
  H = (double complex*)malloc(dim*sizeof(double complex));
  D = (double complex *)malloc(dim*sizeof(double complex));
  P = (double complex *)malloc(dim*sizeof(double complex));
  E  = (double *)malloc(dim*sizeof(double));
  Mu = (double *)malloc(dim*sizeof(double));
  Dis = (double *)malloc(dim*sizeof(double));
  bas = (double *)malloc(dim*sizeof(double));
  Hint = (double *)malloc(dim*sizeof(double));
  // MG
  HMG = (double complex*)malloc(dimMG*sizeof(double complex));
  DMG = (double complex *)malloc(dimMG*sizeof(double complex));
  PMG = (double complex *)malloc(dimMG*sizeof(double complex));
  EMG  = (double *)malloc(dimMG*sizeof(double));
  MuMG = (double *)malloc(dimMG*sizeof(double));
  MuZERO = (double *)malloc(dimMG*sizeof(double));
  DisMG = (double *)malloc(dimMG*sizeof(double));
  basMG = (double *)malloc(dimMG*sizeof(double));
  HintMG = (double *)malloc(dimMG*sizeof(double));

  // Allocate memory for uncoupled arrays
  DD = (double complex*)malloc(dim*sizeof(double complex));
  DA = (double complex*)malloc(dimMG*sizeof(double complex));
  Hint_AD = (double *)malloc(dim*sizeof(double));
  Hint_DA = (double *)malloc(dimMG*sizeof(double));


  // Variables for instantaneous quantities  
  double tr, trMG;
  double complex dipole_moment, dipole_momentMG, dipole_moment_donor, dipole_moment_acceptor;
  FILE *dfp, *dfpMG;
  FILE *popfp, *popMGfp;

  char trash[10000], prefix[1000], dpfn[1000], popfn[1000], absfn[1000], atype[10];

  printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
  printf("                  WELCOME TO NANO RET!\n");
  printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
  printf("\n  ENTER A TITLE FOR YOUR CALCULATION!\n");
  scanf("%s",prefix);
  strcpy(dpfn,"DATA/");
  strcat(dpfn,prefix);
  strcat(dpfn,"_dipoleMoment.dat");

  strcpy(popfn,"DATA/");
  strcat(popfn,prefix);
  strcat(popfn,"_population.txt");

  strcpy(absfn,"DATA/");
  strcat(absfn,prefix);
  strcat(absfn,"_spectra.txt");

  printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
  printf("                  TITLE ENTERED AS '%s'\n",prefix);
  printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
  // Separation vector
  double *r, len;
  r = (double *)malloc(3*sizeof(double));

  int PTYPE;
  FILE *Efp, *Mufp, *Disfp, *EfpMG, *MufpMG, *DisfpMG, *absfp;

  printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
  printf("  WHICH NP WOULD YOU LIKE TO SIMULATE? (DEFAULT IS r=2.6 nm Au)\n");
  printf("\n  FOR GOLD,        TYPE '1' THEN PRESS 'return' TO CONTINUE\n");
  printf("  FOR GOLD@GLASS,    TYPE '2' THEN PRESS 'return' TO CONTINUE\n");
  printf("  FOR SILVER,        TYPE '3' THEN PRESS 'return' TO CONTINUE\n");
  printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
  scanf("%i",&PTYPE);

  if (PTYPE==1) {

    strcpy(atype,"GOLD");
    // INPUT FILES!
    Efp = fopen("MATRICES/Energy_Au.txt","r");
    Mufp = fopen("MATRICES/Dipole_Au.txt","r");
    Disfp = fopen("MATRICES/Dissipation_Au.txt","r");
 
  }
  else if (PTYPE==2) {
    strcpy (atype,"GOLD@GLASS");
    // INPUT FILES!
    Efp = fopen("MATRICES/Energy_SiO2_Au.txt","r");
    Mufp = fopen("MATRICES/Dipole_SiO2_Au.txt","r");
    Disfp = fopen("MATRICES/Dissipation_SiO2_Au.txt","r");

  }
  else if (PTYPE==3) {
    strcpy(atype,"SILVER");
    Efp = fopen("MATRICES/Energy_Ag.txt","r");
    Mufp = fopen("MATRICES/Dipole_Ag.txt","r");
    Disfp = fopen("MATRICES/Dissipation_Ag.txt","r");
  }
  else {

    strcpy(atype,"GOLD");
    // INPUT FILES!
    Efp = fopen("MATRICES/Energy_Au.txt","r");
    Mufp = fopen("MATRICES/Dipole_Au.txt","r");
    Disfp = fopen("MATRICES/Dissipation_Au.txt","r");
  
  }

  printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
  printf("                  NOW GATHERING SEPARATION INFO!\n");
  printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

  printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
  printf("                  ENTER THE SEPARATION BETWEEN ACCEPTOR\n");
  printf("                       AND DONOR IN NANOMETERS!\n");
  printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
  scanf("%lf",&len);

  r[2] = len/bohr_r_nm;
  r[1] = 0.;
  r[0] = 0.; 


  printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
  printf("  WHAT IS THE REFRACTIVE INDEX OF YOUR MEDIUM?\n");
  printf("\n  FOR AIR,      TYPE '1.0' THEN PRESS 'return' TO CONTINUE");
  printf("\n  FOR WATER,    TYPE '1.33' THEN PRESS 'return' TO CONTINUE");
  printf("\n  FOR GLASS,    TYPE '1.5'  THEN PRESS 'return' TO CONTINUE");
  printf("\n  FOR TiO2,     TYPE '2.6'  THEN PRESS 'return' TO CONTINUE\n");
  printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
  scanf("%lf",&RI);

  printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
  printf("\n                     YOU ARE SIMULATING %s NP! \n",atype);
  printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
  //EfpMG = fopen("MATRICES/Energy3LDye.txt","r");
  /*EfpMG = fopen("MATRICES/Energy4LDye.txt","r");
  MufpMG = fopen("MATRICES/Dipole4LDye.txt","r");
  DisfpMG = fopen("MATRICES/Dissipation4LDye.txt","r");
  */
  EfpMG = fopen("MATRICES/Energy_Au.txt","r");
  MufpMG = fopen("MATRICES/Dipole_Au.txt","r");
  DisfpMG = fopen("MATRICES/Dissipation_Au.txt","r");
  // OUTPUT FILES!  MAKE SURE NAMES ARE CONSISTENT
  dfp = fopen(dpfn,"w");
  popfp = fopen(popfn,"w");
  absfp = fopen(absfn,"w");


  // Density matrix element D(i,j) is accessed as D[i*Nlevel+j];
  D[0] = 1. + 0.*I;
  DMG[0] = 1. + 0.*I;
  DD[0] = 1. + 0.*I;
  DA[0] = 1. + 0.*I;

  // NP
  for (int i=1; i<dim; i++){
    D[i] = 0. + 0.*I;
    DD[i] = 0. + 0.*I;
  }
  // MG
  for (int i=1; i<dimMG; i++){
    DMG[i] = 0. + 0.*I;
    DA[i] = 0. + 0.*I;
  }

  // BUILD DM BASIS - this comes into play in Lindblad operator
  // NP
  for (int i=0; i<Nlevel; i++) {
    for (int j=0; j<Nlevel; j++) {
      if (i==j){
        bas[i*Nlevel+j] = 1.0;
      }     
      else{
        bas[i*Nlevel+j] = 0.;
      }
    }
  }
  // MG
  for (int i=0; i<NlevelMG; i++) {
    for (int j=0; j<NlevelMG; j++) {
      if (i==j){
        basMG[i*NlevelMG+j] = 1.0;
      }
      else{
        basMG[i*NlevelMG+j] = 0.;
      }
    }
  }
  
  // Get parameters for NP and MG from files
  double val;
  // NP
  for (int i=0; i<dim; i++) {


       // Read from energy file and store to the energy matrix
       fscanf(Efp,"%lf",&val);
       E[i] = val;

       fscanf(Mufp,"%lf",&val);
       Mu[i] = val;

       fscanf(Disfp,"%lf",&val);
       Dis[i] = val;

       Hint_AD[i] = 0.;
  }
  // MG
  for (int i=0; i<dimMG; i++) {

       fscanf(EfpMG,"%lf",&val);
       EMG[i] = val;

       fscanf(MufpMG,"%lf",&val);
       MuMG[i] = val;
       MuZERO[i] = 0.;
       fscanf(DisfpMG,"%lf",&val);
       DisMG[i] = val;

       Hint_DA[i] = 0.;

  }
 
  /* 
  // Print parameters to screen
  printf("\nE\n");
  PrintRealMatrix(Nlevel, E);
  printf("\nMu\n");
  PrintRealMatrix(Nlevel,Mu);
  printf("\nDis\n");
  PrintRealMatrix(Nlevel,Dis);
  printf("\nBas\n");
  PrintRealMatrix(Nlevel,bas);
  printf("\nDM\n");
  PrintComplexMatrix(Nlevel,D);

  printf("\nEMG\n");
  PrintRealMatrix(NlevelMG, EMG);
  printf("\nMuMG\n");
  PrintRealMatrix(NlevelMG, MuMG);
  printf("\nDisMG\n");
  PrintRealMatrix(NlevelMG, DisMG);
  printf("\nBasMG\n");
  PrintRealMatrix(NlevelMG, basMG);
  printf("\nDMG\n");
  PrintComplexMatrix(NlevelMG, DMG);
  */

  // Get initial dipole moments
  dipole_moment = TrMuD(Nlevel, Mu, D)*mu_au_to_si;
  dipole_momentMG = TrMuD(NlevelMG, MuMG, DMG)*mu_au_to_si;

  FillDFTArray(0, creal(dipole_moment), cimag(dipole_moment), dipole);
  FillDFTArray(0, creal(dipole_momentMG), cimag(dipole_momentMG), dipoleMG);
  FillDFTArray(0, 0., 0., efield);


  //void H_interaction(int dim, double *Hint, double *mu, double dpm, double R) 
  H_interaction(Nlevel, Hint, Mu, creal(dipole_momentMG), r); 
  H_interaction(NlevelMG, HintMG, MuMG, creal(dipole_moment), r);

  for (int i=1; i<numTime; i++) {

    // Update the coupled NP 
    RK3(Nlevel, dt*i, bas, E, Hint, Mu, Dis, D, dt);

    // Update the uncoupled NP
    RK3(Nlevel, dt*i, bas, E, Hint_AD, Mu, Dis, DD, dt);

    // Calculate dipole moment of the coupled NP
    dipole_moment = TrMuD(Nlevel, Mu, D); 
  
    // Calculate the dipole moment of the uncoupled NP
    dipole_moment_donor = TrMuD(Nlevel, Mu, DD);

    // Write dm of coupled NP to fftw array
    FillDFTArray(i, creal(dipole_moment*mu_au_to_si), cimag(dipole_moment*mu_au_to_si), dipole);

    // Write dm of uncoupled NP to fftw array
    FillDFTArray(i, creal(dipole_moment_donor*mu_au_to_si), cimag(dipole_moment_donor*mu_au_to_si), dipole_donor);

    // Calculate interaction matrix between NP and MG: H_int^{np->mg}    
    H_interaction(NlevelMG, HintMG, MuMG, creal(dipole_moment), r);
    
    RK3(NlevelMG, dt*i, basMG, EMG, HintMG, MuMG, DisMG, DMG, dt);
    // Update the coupled MG system
    //RK3_MG(NlevelMG, dt*i, basMG, EMG, HintMG, MuMG, DisMG, DMG, dt);
    //RK3(NlevelMG, dt*i, basMG, EMG, HintMG, MuZERO, DisMG, DMG, dt);
    
    // Update the uncoupled MG system
    //RK3_MG(NlevelMG, dt*i, basMG, EMG, Hint_DA, MuMG, DisMG, DA, dt);
    RK3(NlevelMG, dt*i, basMG, EMG, Hint_DA, MuMG, DisMG, DA, dt);
    // Calculate the dipole moment of coupled MG
    dipole_momentMG = TrMuD(NlevelMG, MuMG, DMG);

    // Calculate the dipole moment of the uncoupled MG
    dipole_moment_acceptor = TrMuD(NlevelMG, MuMG, DA);

    // Write dm of coupled system to FFTW array
    FillDFTArray(i, creal(dipole_momentMG*mu_au_to_si), cimag(dipole_momentMG*mu_au_to_si), dipoleMG);
    
    // Write dm of uncoupled system to FFTW array
    FillDFTArray(i, creal(dipole_moment_acceptor*mu_au_to_si), cimag(dipole_moment_acceptor*mu_au_to_si), dipole_acceptor);

    // Calculate interaction matrix between MG and NP: H_int^{mg->np} 
    H_interaction(Nlevel, Hint, Mu, creal(dipole_momentMG), r); 
   
    // Print population data!
    fprintf(popfp,"\n %f ",dt*i);

    tr=0.;
    trMG = 0.;
    for (int j=0; j<Nlevel; j++) {

      fprintf(popfp," %12.10e",creal(D[j*Nlevel+j]));
      tr+=creal(D[j*Nlevel+j]);

    }
    for (int j=0; j<NlevelMG; j++) {


      fprintf(popfp,"  %12.10e",creal(DMG[j*NlevelMG+j]));
      trMG+=creal(DMG[j*NlevelMG+j]);

    }
    fprintf(popfp," %12.10e",tr);
    fprintf(popfp," %12.10e",trMG);
  
    // Print dipole moment of NP
    fprintf(dfp," %f  %12.10e  %12.10e ",dt*i,creal(dipole_moment),cimag(dipole_moment));
  
    // Print dipole moment of MG
    fprintf(dfp,"%12.10e %12.10e\n",creal(dipole_momentMG),cimag(dipole_momentMG));
 
    FillDFTArray(i,  E_au_to_si*E_Field(dt*i), 0, efield);
  }
 
  printf("  SEPARATION (nm)               SINK_POPULATION            ENERGY_TRANSFERRED (eV)\n");
  printf("  %f                      %12.10f               %12.10f\n",r[2]*bohr_r_nm,creal(DMG[4*1+1]),27.277*EMG[4*2+2]*creal(DMG[4*1+1]));

  // Now to the spectra!
  for (int i=numTime; i<zeropad; i++) {

    FillDFTArray(i, 0., 0., dipole);
    FillDFTArray(i, 0., 0., dipoleMG);
    FillDFTArray(i, 0., 0., efield);
    FillDFTArray(i, 0., 0., dipole_acceptor);
    FillDFTArray(i, 0., 0., dipole_donor);
 
  }
 
  fftw_execute(npp);
  fftw_execute(mgp);
  fftw_execute(efp);
  // FT(dipole_moment_donor) -> alpha_donor
  fftw_execute(donp);
  // FT(dipole_moment_acceptor) -> alpha_acceptor
  fftw_execute(accp); 

  fprintf(absfp, "#Energy(ev) SCAT_CNP SCAT_CMG ABS_CNP ABS_CMG SCAT_NP SCAT_MG ABS_NP ABS_MG\n");
  
  int nfreq = 5001;
  // ~6.48 eV is max energy/ max freq
  double maxfreq = 70*0.08188379587298;
  double df = maxfreq / (nfreq - 1);
  double eps_0 = 1.0 / (4.0 * M_PI);

  // Note that we want to compute R0 here where R0 is the characteristic RET distance
  // R0 = \frac{ 9(ln(10))*\kappa^2 \Phi_D }{ 128 \pi^5 N_A n^4 } \cdot J
  // where
  // J = \int I_D \epsilon_D \lambda^4 d\lambda
    

  // I_D will be the normalized scattering spectrum of the donor
  // \epsilon_D will be the molar extinction coefficient of the acceptor
  // where \epsilon_D = \frac{N_A}{ln(10)} \sigma_A
  // see https://en.wikipedia.org/wiki/Beer%E2%80%93Lambert_law
  
  // \PhiD will be the luminescence QY of the donor
  // N_A is avagadros number
  double PhiD = 1.;
  double NormScat=0.;
  double JInt=0.;
  double domega_au;
  double den;
  double dl_si;

  for (int i=1; i<(numTime+zeropad); i++) {

    // This is omega in atomic units - same as energy in atomic units
    double omega = 2*pi*i/((numTime+zeropad)*dt);
    double omega_p1 = 2*pi*(i+1)/((numTime+zeropad)*dt);
    // d_lambda = 2*pi*c/omega - 2*pi*c/omega_p1
    dl_si = 2*pi*c/(omega*omega_au) - 2*pi*c/(omega_p1*omega_au);
    //printf("  dl_si is %12.10e\n",dl_si);
    
    if (omega * 27.211>maxfreq) break;

    double omega_si = omega*omega_au;
    double eev = omega*27.211;
    double lambda_nm = 1240./eev;
    double lambda_si = lambda_nm*1e-9;

    double k = omega_si/2.99792458e+8;
    double pre_scat = k*k*k*k/(6*pi*8.854187e-12*8.854187e-12*pow(RI,4.)); 
    double pre_abs = k/(8.854187e-12*RI*RI);
   
    double npr = nps[i][0]/numTime;
    double npi = nps[i][1]/numTime;

    double mgr = mgs[i][0]/numTime;
    double mgi = mgs[i][1]/numTime;

    double efr = efs[i][0]/numTime;
    double efi = efs[i][1]/numTime;

    double donr = alpha_donor[i][0]/numTime;
    double doni = alpha_donor[i][1]/numTime;

    double accr = alpha_acceptor[i][0]/numTime;
    double acci = alpha_acceptor[i][1]/numTime;

    double complex alphaNP = (npr+I*npi)/(efr+I*efi);
    double complex alphaMG = (mgr+I*mgi)/(efr+I*efi);

    double complex Adonor  = (donr+I*doni)/(efr+I*efi);
    double complex Aaccept = (accr+I*acci)/(efr+I*efi);

    double sig_scat_NP = pre_scat * creal(alphaNP*conj(alphaNP));
    double sig_scat_MG = pre_scat * creal(alphaMG*conj(alphaMG));
    double sig_abs_NP = pre_abs * cimag(alphaNP);
    double sig_abs_MG = pre_abs * cimag(alphaMG);

    // Cross sections of uncoupled donor
    double sig_scat_don = pre_scat * creal(Adonor*conj(Adonor));
    double sig_abs_don  = pre_abs * cimag(Adonor);

    // To normalize donor spectrum
    NormScat += sig_scat_don * dl_si;    
    // Cross sections of uncoupled acceptor
    double sig_scat_acc = pre_scat * creal(Aaccept*conj(Aaccept));
    double sig_abs_acc = pre_abs *  cimag(Aaccept);

    // Contribution to J integral
    JInt += sig_scat_don*sig_abs_acc*NA/log(10.)*pow(lambda_si,4.)*dl_si; 

    // Going to print absorption and scattering cross section in m^2
    fprintf(absfp, "%12.10e %12.10e %12.10e %12.10e %12.10e %12.10e %12.10e %12.10e %12.10e\n",eev,sig_scat_NP, sig_scat_MG, sig_abs_NP, sig_abs_MG, sig_scat_don, sig_scat_acc, sig_abs_don, sig_abs_acc);
  }
  //printf("  NormScat is %12.10e\n",NormScat);
  //printf("  Jint before norm is %12.10e\n",JInt);
  JInt /= NormScat;
  //printf("  Jint after norm is %12.10e\n",JInt);
// R0 = \frac{ 9(ln(10))*\kappa^2 \Phi_D }{ 128 \pi^5 N_A n^4 } \cdot J
  double R0 = 9*log(10)/(128.*pow(pi,5.)*NA*RI*RI*RI*RI)*JInt;
  //printf("  R0 is %12.10e\n",pow(R0,(1./6.)));

  fclose(absfp);
  
  return 0;
}


void PrintRealMatrix(int dim, double *M) {

  printf("\n");
  for (int i=0; i<dim; i++) {

    for (int j=0; j<dim; j++) {

      printf(" %f ",M[i*dim+j]);

    }
    printf("\n");
  }

  printf("\n");
}

void PrintComplexMatrix(int dim, double complex *M) {
 
  printf("\n");
  for (int i=0; i<dim; i++) {

    for (int j=0; j<dim; j++) {

      printf(" (%12.10e,%12.10e) ",creal(M[i*dim+j]),cimag(M[i*dim+j]));

    }
    printf("\n");
  }
  printf("\n");
}

void RK3(int Nlevel, double time, double *bas, double *E, double *Hint, double *Mu, double *Dis, double complex *D, double dt) {

  int i, j;
  double complex *D_dot, *D2, *D3, *D_np1, *k1, *k2, *k3;
  double complex *H, *LD;  // Contribution to Ddot from Lindblad dissipation
  double *gamma;
  int dim = Nlevel*Nlevel; 
  double Efield;

  D_dot = (double complex *)malloc(dim*sizeof(double complex));
  D2    = (double complex *)malloc(dim*sizeof(double complex));
  D3    = (double complex *)malloc(dim*sizeof(double complex));
  D_np1 = (double complex *)malloc(dim*sizeof(double complex));
  k1    = (double complex *)malloc(dim*sizeof(double complex));
  k2    = (double complex *)malloc(dim*sizeof(double complex));
  k3    = (double complex *)malloc(dim*sizeof(double complex));
  H     = (double complex *)malloc(dim*sizeof(double complex));
  LD    = (double complex *)malloc(dim*sizeof(double complex));
  gamma = (double *)malloc(Nlevel*sizeof(double));
  
  // Must zero out all elements of these arrays
  for (i=0; i<dim; i++) {
    D_dot[i] = 0. + 0.*I;
    D2[i] = 0. + 0.*I;
    D3[i] = 0. + 0.*I;
    D_np1[i] = 0. + 0.*I;
    k1[i] = 0. + 0.*I;
    k2[i] = 0. + 0.*I;
    k3[i] = 0. + 0.*I;
    H[i] = 0. + 0.*I;
   
  }

  for (i=0; i<Nlevel; i++) {
    gamma[i] = Dis[i*Nlevel+i];
  }
  Efield = E_Field(time);

  // Compute full Hamiltonian at current time t 
  for (i=0; i<dim; i++) {


      H[i] = E[i] + Hint[i] - Efield*Mu[i]; // - I*Dis[i];

  } 

  //PrintComplexMatrix(Nlevel, H);

  // Get dPsi(n)/dt at initial time!
  // Two main changes needed to couple the molecule and nanoparticle:
  // (1) Liouville function needs to include H_interaction
  // (2) We need to use Liouville/L_Diss to update both the molecule and the nanoparticle density matrix
  Liouville(Nlevel, H, D, D_dot);
  L_Diss(Nlevel, gamma, D, bas, LD);
  //PrintComplexMatrix(Nlevel, D);
  //PrintComplexMatrix(Nlevel, D_dot);


  // Compute approximate wfn update with Euler step
  for (i=0; i<dim; i++) {
    k1[i] = dt*(D_dot[i]+LD[i]);
    D2[i] = D[i] + k1[i]/2.;
  }

  // Update Field!
  Efield = E_Field(time+dt/2.);

  // Compute full Hamiltonian at partially updated time t 
  for (i=0; i<dim; i++) {

      H[i] = E[i] + Hint[i] - Efield*Mu[i]; // - I*Dis[i];

  }

  //PrintComplexMatrix(Nlevel, H);
  // Get dPsi(n+k1/2)/dt
  Liouville (Nlevel, H, D2, D_dot);
  L_Diss(Nlevel, gamma, D2, bas, LD);
  
  // Compute approximate wfn update with Euler step
  for (i=0; i<dim; i++) {
    k2[i] = dt*(D_dot[i] + LD[i]);
    D3[i] = D[i] + k2[i]/2.;
  }

  // Get dPsi(n+k2/2)/dt
  Liouville (Nlevel, H, D3, D_dot);
  L_Diss(Nlevel, gamma, D3, bas, LD);

  // Compute approximate update with Euler step
  for (i=0; i<dim; i++) {
    k3[i] = dt*(D_dot[i] + LD[i]);
    D_np1[i] = D[i] + k1[i]/6. + 2.*k2[i]/3. + k3[i]/6.;
    D[i] = D_np1[i];
}


  free(D_dot);
  free(D2);
  free(D3);
  free(D_np1);
  free(k1);
  free(k2);
  free(k3);
  free(H);
  free(LD);
  free(gamma);
}


void RK3_MG(int Nlevel, double time, double *bas, double *E, double *Hint, double *Mu, double *Dis, double complex *D, double dt) {

  int i, j;
  double complex *D_dot, *D2, *D3, *D_np1, *k1, *k2, *k3;
  double complex *H, *LD_rad1, *LD_rad2, *LD_nonrad1, *LD_nonrad2;  // Contribution to Ddot from Lindblad dissipation
  double *gamma;
  int dim = Nlevel*Nlevel; 
  double Efield;

  D_dot = (double complex *)malloc(dim*sizeof(double complex));
  D2    = (double complex *)malloc(dim*sizeof(double complex));
  D3    = (double complex *)malloc(dim*sizeof(double complex));
  D_np1 = (double complex *)malloc(dim*sizeof(double complex));
  k1    = (double complex *)malloc(dim*sizeof(double complex));
  k2    = (double complex *)malloc(dim*sizeof(double complex));
  k3    = (double complex *)malloc(dim*sizeof(double complex));
  H     = (double complex *)malloc(dim*sizeof(double complex));
  LD_rad1    = (double complex *)malloc(dim*sizeof(double complex));
  LD_nonrad1 = (double complex *)malloc(dim*sizeof(double complex));
  LD_rad2    = (double complex *)malloc(dim*sizeof(double complex));
  LD_nonrad2 = (double complex *)malloc(dim*sizeof(double complex));
  gamma = (double *)malloc(Nlevel*sizeof(double));
  
  // Must zero out all elements of these arrays
  for (i=0; i<dim; i++) {
    D_dot[i] = 0. + 0.*I;
    D2[i] = 0. + 0.*I;
    D3[i] = 0. + 0.*I;
    D_np1[i] = 0. + 0.*I;
    k1[i] = 0. + 0.*I;
    k2[i] = 0. + 0.*I;
    k3[i] = 0. + 0.*I;
    H[i] = 0. + 0.*I;
   
  }

  for (i=0; i<Nlevel; i++) {
    gamma[i] = Dis[i*Nlevel+i];
  }
  // get the total dissipation rates associated with excited-state 1 and 2
  // For Malachite Green (4-level system), these are states 3 and 4, respectively
  // going to divide dissipation equally among radiative and non-radiative channels
  // For Malachite green, radiative transitions are from state 4->1 and 3->1
  // and nonradiative transitions are from 4->2 and 3->2
  double g_val1 = gamma[Nlevel-2];
  double g_val2 = gamma[Nlevel-1];
  Efield = E_Field(time);

  // Compute full Hamiltonian at current time t 
  for (i=0; i<dim; i++) {


      H[i] = E[i] + Hint[i] - Efield*Mu[i]; // - I*Dis[i];

  } 

  //PrintComplexMatrix(Nlevel, H);

  // Get dPsi(n)/dt at initial time!
  // Two main changes needed to couple the molecule and nanoparticle:
  // (1) Liouville function needs to include H_interaction
  // (2) We need to use Liouville/L_Diss to update both the molecule and the nanoparticle density matrix
  Liouville(Nlevel, H, D, D_dot);

  // Radiative transition from state 3->1
  L_Diss_MG(Nlevel, g_val1/2., D, bas, LD_rad1, 2, 0);
  // Radiative transition from state 4->1
  L_Diss_MG(Nlevel, g_val2/2., D, bas, LD_rad2, 3, 0);
  // non-radiative transition from state 3->2
  L_Diss_MG(Nlevel, g_val1/2., D, bas, LD_nonrad1, 2, 1);
  // non-radiative transition from state 4->2
  L_Diss_MG(Nlevel, g_val2/2., D, bas, LD_nonrad2, 3, 1);
  //PrintComplexMatrix(Nlevel, D);
  //PrintComplexMatrix(Nlevel, D_dot);


  // Compute approximate wfn update with Euler step
  for (i=0; i<dim; i++) {
    k1[i] = dt*(D_dot[i]+LD_rad1[i]+LD_rad2[i]+LD_nonrad1[i]+LD_nonrad2[i]);
    D2[i] = D[i] + k1[i]/2.;
  }

  // Update Field!
  Efield = E_Field(time+dt/2.);

  // Compute full Hamiltonian at partially updated time t 
  for (i=0; i<dim; i++) {

      H[i] = E[i] + Hint[i] - Efield*Mu[i]; // - I*Dis[i];

  }

  //PrintComplexMatrix(Nlevel, H);
  // Get dPsi(n+k1/2)/dt
  Liouville (Nlevel, H, D2, D_dot);
  // Radiative transition from state 3->1
  L_Diss_MG(Nlevel, g_val1/2., D2, bas, LD_rad1, 2, 0);
  // Radiative transition from state 4->1
  L_Diss_MG(Nlevel, g_val2/2., D2, bas, LD_rad2, 3, 0);
  // non-radiative transition from state 3->2
  L_Diss_MG(Nlevel, g_val1/2., D2, bas, LD_nonrad1, 2, 1);
  // non-radiative transition from state 4->2
  L_Diss_MG(Nlevel, g_val2/2., D2, bas, LD_nonrad2, 3, 1);

  
  // Compute approximate wfn update with Euler step
  for (i=0; i<dim; i++) {
    k2[i] = dt*(D_dot[i] + LD_rad1[i]+LD_rad2[i]+LD_nonrad1[i]+LD_nonrad2[i]);
    D3[i] = D[i] + k2[i]/2.;
  }

  // Get dPsi(n+k2/2)/dt
  Liouville (Nlevel, H, D3, D_dot);
  // Radiative transition from state 3->1
  L_Diss_MG(Nlevel, g_val1/2., D3, bas, LD_rad1, 2, 0);
  // Radiative transition from state 4->1
  L_Diss_MG(Nlevel, g_val2/2., D3, bas, LD_rad2, 3, 0);
  // non-radiative transition from state 3->2
  L_Diss_MG(Nlevel, g_val1/2., D3, bas, LD_nonrad1, 2, 1);
  // non-radiative transition from state 4->2
  L_Diss_MG(Nlevel, g_val2/2., D3, bas, LD_nonrad2, 3, 1);

  // Compute approximate update with Euler step
  for (i=0; i<dim; i++) {
    k3[i] = dt*(D_dot[i] + LD_rad1[i]+LD_rad2[i]+LD_nonrad1[i]+LD_nonrad2[i]);
    D_np1[i] = D[i] + k1[i]/6. + 2.*k2[i]/3. + k3[i]/6.;
    D[i] = D_np1[i];
}


  free(D_dot);
  free(D2);
  free(D3);
  free(D_np1);
  free(k1);
  free(k2);
  free(k3);
  free(H);
  free(LD_rad1);
  free(LD_nonrad1);
  free(LD_rad2);
  free(LD_nonrad2);
  free(gamma);
}




void FormDM(int dim, double complex *C, double complex *Ct, double complex *DM) {

  for (int i=0; i<dim; i++) {

    for (int j=0; j<dim; j++) {

      DM[i*dim+j] = C[i]*Ct[j];

    }

  }

}

void Liouville(int dim, double complex *H, double complex *D, double complex *P) {


  // write code here!
  for (int i=0; i<dim; i++) {

    for (int j=0; j<dim; j++) {


      double complex sum2 = 0.+0.*I;
      double complex sum1 = 0.+0.*I;
      for (int k=0; k<dim; k++) {

        sum1 -= H[i*dim+k]*D[k*dim+j]*I;
        sum2 += D[i*dim+k]*H[k*dim+j]*I;
      }
      P[i*dim+j] = sum1 + sum2;
    }
  }
}



void AntiCommutator(int dim, double *H, double complex *D, double complex *P) {

// write code here!
for (int i=0; i<dim; i++) {

  for (int j=0; j<dim; j++) {


    double complex sum2 = 0.+0.*I;
    double complex sum1 = 0.+0.*I;
    for (int k=0; k<dim; k++) {

      sum1 += H[i*dim+k]*D[k*dim+j];
      sum2 += D[i*dim+k]*H[k*dim+j];
    }
    P[i*dim+j] = sum1 + sum2;
    //printf(" Pb[%i][%i] is %f %f\n",i,j,creal(sum1-sum2),cimag(sum1-sum2));
}
}
}


double E_Field(double time) {

  double Ef;
  double tau = 75.;

  //Ef = 0.01*sin(pi*time/tau)*sin(pi*time/tau)*exp(-0.005*time)*(sin(0.07423*time)+sin(0.1*time)+sin(0.5*time));
  if (time<tau) {

  Ef = 0.0001*sin(time*pi/tau)*sin(time*pi/tau)*sin(0.07423*time);

  }
  else Ef = 0.;
    
  return Ef;


}

void L_Diss(int Nlevel, double *gamma, double complex *D, double *bas, double complex *P) {

  int i, j, k;
  double *temp_bas, *g_bas;
  double complex *temp_t1, *temp_t2, *LD;
  temp_bas = (double *)malloc(Nlevel*Nlevel*sizeof(double));
  g_bas    = (double *)malloc(Nlevel*Nlevel*sizeof(double));
  temp_t1  = (double complex *)malloc(Nlevel*Nlevel*sizeof(double complex));
  temp_t2  = (double complex *)malloc(Nlevel*Nlevel*sizeof(double complex));
  LD       = (double complex *)malloc(Nlevel*Nlevel*sizeof(double complex));
 
  double gk;
  // Form |g><g| matrix
  for (i=0; i<Nlevel; i++) {
    for (j=0; j<Nlevel; j++) {
      g_bas[i*Nlevel+j] = bas[0*Nlevel+i]*bas[j*Nlevel+0];
      LD[i*Nlevel+j] = 0. + 0.*I;
    }
  }
  //printf("  |g><g|  \n");
  //PrintRealMatrix(Nlevel, g_bas);

  for (k=1; k<Nlevel; k++) {

    gk = gamma[k];
    for (i=0; i<Nlevel; i++) {

      for (j=0; j<Nlevel; j++) {

        temp_bas[i*Nlevel+j] = bas[k*Nlevel+i]*bas[j*Nlevel+k];
        temp_t1[i*Nlevel+j] = 2*D[k*Nlevel+k]*g_bas[i*Nlevel+j];
      }
    }
   // printf("   |%i><%i| \n",k,k);
   // PrintRealMatrix(Nlevel, temp_bas);

    AntiCommutator(Nlevel, temp_bas, D, temp_t2);
    for (i=0; i<Nlevel; i++) {
      for (j=0; j<Nlevel; j++) {
        LD[i*Nlevel+j] += gk*temp_t1[i*Nlevel+j] - gk*temp_t2[i*Nlevel+j];
      }
    }
 }
 for (i=0; i<Nlevel; i++) {
   for (j=0; j<Nlevel; j++) {
     P[i*Nlevel+j] = LD[i*Nlevel+j];
   }
 }

 //PrintComplexMatrix(Nlevel, P);

 free(temp_bas);
 free(g_bas);
 free(temp_t1);
 free(temp_t2);
 free(LD);
}


void L_Diss_MG(int Nlevel, double gamma, double complex *D, double *bas, double complex *P, int ex, int dex) {

  int i, j, k;
  double *temp_bas, *g_bas;
  double complex *temp_t1, *temp_t2, *LD;
  temp_bas = (double *)malloc(Nlevel*Nlevel*sizeof(double));
  g_bas    = (double *)malloc(Nlevel*Nlevel*sizeof(double));
  temp_t1  = (double complex *)malloc(Nlevel*Nlevel*sizeof(double complex));
  temp_t2  = (double complex *)malloc(Nlevel*Nlevel*sizeof(double complex));
  LD       = (double complex *)malloc(Nlevel*Nlevel*sizeof(double complex));
 
  double gk;
  // Form |g><g| matrix
  for (i=0; i<Nlevel; i++) {
    for (j=0; j<Nlevel; j++) {
      g_bas[i*Nlevel+j] = bas[dex*Nlevel+i]*bas[j*Nlevel+dex];
      LD[i*Nlevel+j] = 0. + 0.*I;
    }
  }
  //printf("  |g><g|  \n");
  //PrintRealMatrix(Nlevel, g_bas);


    gk = gamma;
    for (i=0; i<Nlevel; i++) {

      for (j=0; j<Nlevel; j++) {

        temp_bas[i*Nlevel+j] = bas[ex*Nlevel+i]*bas[j*Nlevel+ex];
        temp_t1[i*Nlevel+j] = 2*D[ex*Nlevel+ex]*g_bas[i*Nlevel+j];
      }
    }
   // printf("   |%i><%i| \n",k,k);
   // PrintRealMatrix(Nlevel, temp_bas);

    AntiCommutator(Nlevel, temp_bas, D, temp_t2);
    for (i=0; i<Nlevel; i++) {
      for (j=0; j<Nlevel; j++) {
        LD[i*Nlevel+j] += gk*temp_t1[i*Nlevel+j] - gk*temp_t2[i*Nlevel+j];
      }
    }
 for (i=0; i<Nlevel; i++) {
   for (j=0; j<Nlevel; j++) {
     P[i*Nlevel+j] = LD[i*Nlevel+j];
   }
 }

 //PrintComplexMatrix(Nlevel, P);

 free(temp_bas);
 free(g_bas);
 free(temp_t1);
 free(temp_t2);
 free(LD);
}

void Fourier(double complex *dm, int n, double dt, double complex *ftout, double *freqvec){
  //FILE *fp;
  //fp = fopen("Absorption_SpectrumAu.txt","w");
  double wmin=0.5*0.07;
  double wmax=2*0.07;
  int maxk = 500;
  double dw = (wmax-wmin)/maxk;
  double time;
 
  for (int k = 0; k <=maxk; k++) {
    double sumreal = 0;
    double sumimag = 0;
    double  w = wmin+k*dw;

    for (int t = 0; t < n; t++){
      time = dt*t;
      double angle = time*w;
      sumreal += creal(cexp(-I*angle)*dm[t])*dt;
      sumimag += cimag(cexp(-I*angle)*dm[t])*dt;
      //sumreal += creal(dm[t]) * cos(angle) + cimag(dm[t]) * sin(angle);
      //sumimag  += creal(dm[t]) * sin(angle) + cimag(dm[t]) * cos(angle);
    }
    ftout[k] = sumreal + sumimag*I;
    // Energy in eV
    freqvec[k] = w*27.2114;
    //fprintf(fp," %12.10e  %12.10e\n",w*(27.2114),sumreal*sumreal+sumimag*sumimag);
  }
  //fclose(fp);
}

double complex TrMuD(int Nlevel, double *Mu, double complex *D) {
  double complex tr = 0. + 0.*I;
  for (int i=0; i<Nlevel; i++) {

    double complex sum = 0. + 0.*I;
    for (int k=0; k<Nlevel; k++) {

      sum += Mu[i*Nlevel+k]*D[k*Nlevel+i];

    }
    tr += sum;

  } 

  return tr;

}

// added 1/n^2 dependence to dipole-dipole potential where n is the medium refractive index
void H_interaction(int dim, double *Hint, double *mu, double dpm, double *R) {
  
  int i; 
 // double *tmp1, *tmp2;
  double oer2, oer3;
  double scal_R = sqrt(R[0]*R[0] + R[1]*R[1] + R[2]*R[2]);
 
  oer2 = pow(scal_R,-2.);
  oer3 = pow(scal_R,-3.);
 
  // Write code between here!
 
 for (i=0; i<dim*dim; i++){

   // Very important!  Currently assuming z-polarized light, so only the <mu>_z and mu_z terms
   //                  are non-zero, hence we consider only R[2]*mu and <mu>*R[2] 
   Hint[i] = (1./(RI*RI))*oer3*(dpm*mu[i] -3*R[2]*mu[i]*R[2]*dpm*oer2);
 } 

}

void FillDFTArray(int fftw_iter, double real_val, double imag_val, fftw_complex* ar) {

  ar[fftw_iter][REAL] = real_val;
  ar[fftw_iter][IMAG] = imag_val;

}

// Computes the Frobenius norm of the difference between the current density matrix and the initial density matrix
double D_Error(int dim, double complex *D) {

  double complex e;
  double complex one = 1. + 0.*I;
  double complex zero = 0. + 0.*I;
  double FN;

  e = 0.+0*I;
 
  e = (one - D[0])*conj(one-D[0]);

  for (int i=1; i<dim; i++) {

    e += (zero - D[i])*conj(zero-D[i]);  

  }

  FN = creal(csqrt(e));

  return FN;
}

