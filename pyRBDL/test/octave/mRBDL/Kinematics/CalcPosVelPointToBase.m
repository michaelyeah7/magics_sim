function [pos, vel] = CalcPosVelPointToBase(model, q, qdot, idbody, idbase, tarpoint)
%CalcPosVelPointToBase - Calculate the position and velocity of a point reletive to base
%
% Syntax: [pos, vel] = CalcPosVelPointToBase(q, qdot, idbody, tarpoint)
%
% Long description

    pos_body = CalcBodyToBaseCoordinates(model, q, idbody, tarpoint);
    vel_body = CalcPointVelocity(model, q, qdot, idbody, tarpoint);

    pos_base = CalcBodyToBaseCoordinates(model, q, idbase, zeros(3, 1));
    vel_base = CalcPointVelocity(model, q, qdot, idbase, zeros(3, 1));

    pos = pos_body - pos_base;
    vel = vel_body - vel_base;
    
end