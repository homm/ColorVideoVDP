import torch
import torch.nn.functional as F
import pycvvdp.utils as utils

from pycvvdp.interp import interp1q, batch_interp1d

class castleCSF:

    def __init__(self, csf_version, device, config_paths=[]):
        self.device = device
        csf_lut_file = utils.config_files.find( f"csf_lut_{csf_version}.json", config_paths )
        csf_lut = utils.json2dict(csf_lut_file)

        log_L_bkg = torch.log10( torch.as_tensor(csf_lut["L_bkg"], device=device) )
        log_rho = torch.log10( torch.as_tensor(csf_lut["rho"], device=device) )
        self.log_L_bkg = torch.linspace(log_L_bkg[0], log_L_bkg[-1], (log_L_bkg.numel()-1)*4+1, device=device)
        self.log_rho = torch.linspace(log_rho[0], log_rho[-1], (log_rho.numel()-1)*4+1, device=device)
        self.omega = csf_lut["omega"]

        self.S = []
        for oo in range(2): # For each temp frequency
            self.S.append([])
            ch_num = 3 if oo==0 else 1
            for cc in range(ch_num):
                field_name = f"o{self.omega[oo]}_c{cc+1}"
                logS = torch.as_tensor(csf_lut[field_name], device=device)[None, None]
                logS = F.interpolate(logS, size=(self.log_L_bkg.numel(), self.log_rho.numel()), mode="bilinear", align_corners=True)
                self.S[oo].append( 10**logS[0, 0] )

        self.S_rho = {}


    def sensitivity(self, rho, omega, logL_bkg, cc, sigma):
        # rho - spatial frequency
        # omega - temporal frequency
        # L_bkg - background luminance
        # sigma - radius of spatial integration (Gaussian envelope)

        # Which LUT to use
        oo = 0 if omega==0 else 1
        S_lut = self.S[oo][cc]

        # First interpolate between spatial frequencies rho
        rho_str = f"o{oo}_c{cc}_rho{rho}"
        if rho_str in self.S_rho: # Check if it is cached
            S_r = self.S_rho[rho_str]
        else:
            N = self.log_L_bkg.numel()
            S_r = batch_interp1d(torch.log10(torch.as_tensor(rho, device=self.device, dtype=torch.float32)).expand(N), self.log_rho, S_lut)
            self.S_rho[rho_str] = S_r

        # Then, interpolate across luminance levels
        S = interp1q( self.log_L_bkg, S_r, logL_bkg )

        return S

    def update_device( self, device ):
        self.device = device
        self.log_L_bkg = self.log_L_bkg.to(device)
        self.log_rho = self.log_rho.to(device)

        for oo in range(2): # For each temp frequency
            ch_num = 3 if oo==0 else 1
            for cc in range(ch_num):
                self.S[oo][cc] = self.S[oo][cc].to(device)
