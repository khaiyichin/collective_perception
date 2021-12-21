function heatmap_matrix = import_heatmap_data(filename, numCols, dataLines)
%IMPORTFILE Import data from a text file
%  HEATMAP_MATRIX = IMPORTFILE(FILENAME, NUMCOLS) reads NUMCOLS columns
%  from text file FILENAME for the default selection.  Returns the
%  numeric data.
%
%  HEATMAP_MATRIX = IMPORTFILE(FILE, NUMCOLS, DATALINES) reads NUMCOLS
%  columns for the specified row interval(s) of text file FILENAME. Specify
%  DATALINES as a positive scalar integer or a N-by-2 array of positive
%  scalar integers for dis-contiguous row intervals.
%
%  Example:
%  heatmap_matrix = importfile("C:\Users\khaiy\Documents\collective_consensus\scripts\python\data\heatmap_mean16122021_114721_50_1000.csv", [1, Inf]);
%
%  See also READTABLE.
%
% Auto-generated by MATLAB on 16-Dec-2021 11:55:37

%% Input handling

% If dataLines is not specified, define defaults
if nargin < 3
    dataLines = [1, Inf];
end

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", numCols);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
types = repmat({'double'}, 1, numCols);

% opts.VariableNames = ["e05", "VarName2", "VarName3", "VarName4", "VarName5", "VarName6", "VarName7", "VarName8", "VarName9", "VarName10"];
% opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

opts.VariableNames = types;
opts.VariableTypes = types;

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
heatmap_matrix = readtable(filename, opts);

%% Convert to output type
heatmap_matrix = table2array(heatmap_matrix);
end