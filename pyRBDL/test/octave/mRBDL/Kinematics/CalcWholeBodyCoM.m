function Pcom = CalcWholeBodyCoM(model, q)
%CalcWholeBodyCoM - Calculate whole body's CoM position in world frame
%
% Syntax: Pcom = CalcWholeBodyCoM(q)
%
% Long description


    num = max(size(model.idcomplot,1), size(model.idcomplot,2));


    % Calcualte link's CoM in world frame
    CoM = [];
    Clink = zeros(3,1);
    for i=1:num
        Clink = CalcBodyToBaseCoordinates(model, q, model.idcomplot(i), model.CoM{i});
        CoM = [CoM, Clink]; 
    end

    % Calculate whole CoM in world frame    
    C = zeros(3,1);
    M = 0;
    for i=1:num
        C = C + CoM(:, i) * model.Mass{i};
        M = M + model.Mass{i};
    end
    Pcom = C/M;
end