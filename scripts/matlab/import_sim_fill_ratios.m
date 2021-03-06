function des_f_avg_f_matrix = import_sim_fill_ratios(filename, verificationFillRatios, dataLines)
%IMPORT_SIM_FILL_RATIOS Import data from a text file
%  DES_F_AVG_F_MATRIX = IMPORT_SIM_FILL_RATIOS(FILENAME, 
%  VERIFICATIONFILLRATIOS) reads data from text file FILENAME for the 
%  default selection and verifies it with VERIFICATIONFILLRATIOS.  Returns 
%  the numeric data.
%
%  DES_F_AVG_F_MATRIX = IMPORT_SIM_FILL_RATIOS(FILE,
%  VERIFICATIONFILLRATIOS, DATALINES) reads data for the specified row 
%  interval(s) of text file FILENAME and verifies it with 
%  VERIFICATIONFILLRATIOS. Specify DATALINES as a positive scalar integer 
%  or a N-by-2s array of positive scalar integers for dis-contiguous row 
%  intervals.
%
%  Example:
%  des_f_avg_f_matrix = import_sim_fill_ratios("C:\Users\khaiy\Documents\collective_consensus\scripts\python\data\notable_data\01062022_095829_c200_o100_p55-2-100_f5-5-95\des_f_avg_f_c200_o100_p55-2-100_f5-5-95.csv", [1, Inf]);
%
%  See also READTABLE.
%
% Auto-generated by MATLAB on 06-Jan-2022 11:04:08

%% Input handling

% If dataLines is not specified, define defaults
if nargin < 3
    dataLines = [1, Inf];
end

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 2);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["des_f", "avg_f"];
opts.VariableTypes = ["double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
candidate = readtable(filename, opts);

%% Convert to output type
candidate = table2array(candidate);

%% Ensure that the correct set of fill ratios are read
verificationFillRatios = reshape( verificationFillRatios, [ 1, length( candidate(:, 1) ) ] );
importedFillRatios = reshape( candidate(:, 1), [ 1, length( candidate(:, 1) ) ] ); % the first column of the imported file is the desired fill ratio

% Check to see the two vectors are similar within 1e-10
if all( abs( importedFillRatios - verificationFillRatios ) < 1e-10 )
    des_f_avg_f_matrix = candidate;
else
    error('The imported fill ratios do not match the desired fill ratios.')
end

end