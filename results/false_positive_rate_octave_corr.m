function [] = false_positive_rate_octave_corr(dataset)

    for n = 0:999
        n = num2str(n);
        Vcon1 = spm_read_vols(spm_vol(fullfile('/Volumes','Expansion','pipeline_transition', 'figures', ['ER_', dataset, '_VS_', dataset], 
            'final_results_group_comparison', 'l2_analysis', '_contrast_rh', ['_n_', n], 'spmT_0001_thresholded_FWE.nii')));
        V2 = spm_read_vols(spm_vol(fullfile('/Volumes','Expansion','pipeline_transition', 'figures', ['ER_', dataset, '_VS_', dataset], 
            'final_results_group_comparison', 'l2_analysis', '_contrast_rh', ['_n_', n], 'mask.nii')));
        Vcon1 = Vcon1(:);
        Vcon1 = Vcon1(~isnan(Vcon1));
        V2 = V2(:);
        V2 = V2((V2>0));
        fract = sum(Vcon1>0)/length(V2);
        
        save('-mat7-binary',fullfile('/Users', 'egermani', 'Documents', 'pipeline_transition', 'figures', ['ER_', dataset, '_VS_', dataset],['FPR_FWE_',n,'.mat']),'fract')

        
        Vcon2 = spm_read_vols(spm_vol(fullfile('/Volumes','Expansion','pipeline_transition', 'figures', ['ER_', dataset, '_VS_', dataset], 
            'final_results_group_comparison', 'l2_analysis', '_contrast_rh', ['_n_', n], 'spmT_0002_thresholded_FWE.nii')));

        Vcon2 = Vcon2(:);
        Vcon2 = Vcon2(~isnan(Vcon2));
        fract2 = sum(Vcon2>0)/length(V2);
        
        save('-mat7-binary',fullfile('/Users', 'egermani', 'Documents', 'pipeline_transition', 'figures', ['ER_', dataset, '_VS_', dataset],['FPR_FWE_2_',n,'.mat']),'fract2')
    end

    for n=0:999
        n=num2str(n)
        a = load(fullfile('/Users', 'egermani', 'Documents', 'pipeline_transition', 'figures', ['ER_', dataset, '_VS_', dataset],['FPR_FWE_',n,'.mat']))
        Lfract=[Lfract,((a.fract)>0)*1]
        Lmean = [Lmean,mean(Lfract)]
    end
    mean=Lmean(1000)
    save('-mat7-binary',fullfile('/Users', 'egermani', 'Documents', 'pipeline_transition', 'figures', ['ER_', dataset, '_VS_', dataset], ['mean_hand_FWE.mat']),'mean')
end