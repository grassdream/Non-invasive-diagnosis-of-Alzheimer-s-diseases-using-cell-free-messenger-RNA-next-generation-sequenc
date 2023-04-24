cd D:\Bingo\项目\data\各个分类数据结果 
clear;
% file_name = "238mrnaRF.csv"
file_names = ["238lncrnaRF.csv" "238mirnaRF.csv" "238SNPRF.csv" "238WEIbilireads.csv"];
for i=1:length(file_names)
    file_name = file_names(i)
    data = rows2vars(readtable(file_name));
    AD_idx = find(cell2mat(table2array(data(2,2:end)))==1)+1;
    NCI_idx = find(cell2mat(table2array(data(2,2:end)))==0)+1;
    gene = data(3:end,1);
    AD = cell2mat(table2array(data(3:end,AD_idx)));
    NCI = cell2mat(table2array(data(3:end,NCI_idx)));
    [p,t]=mattest(AD,NCI);
    res=[gene array2table(t) array2table(p)];
    res=sortrows(res,3); 
    % 按照t-test的p-value从小到大排序，这样，排在前面的基因是最差异表达的基因
    writetable(res(:,1),'ttest_deg_'+file_name);
end
