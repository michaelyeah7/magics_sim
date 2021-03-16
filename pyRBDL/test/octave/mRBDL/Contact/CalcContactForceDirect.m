function [flcp, fqp] = CalcContactForceDirect(model, q, qdot, tau, flag_contact, nf)
%CalcContactForceDirect - Calculate contact force by linear solution
%
% Syntax: [flcp, fqp] = CalcContactForceDirect(q, qdot, tau, flag_contact)
%
% flcp : contact force in joint space solved from LCP 
% fqp  : contact force in world space 
% The linear solution is a special solution of LCP, where:
% flcp = Jc' * fqp
% fqp = -d/M
% d = Jc * H^-1 * (tau - C) + Jcdot*qdot
% M = Jc * H^-1 * Jc'

    flag_recalc = 1;

    while flag_recalc
        if sum(flag_contact)==0
            fqp = zeros(model.NC*nf, 1);
            flcp = zeros(model.NB, 1);
            break;
        end

        % Calculate contact force
        Jc = CalcContactJacobian(model, q, flag_contact, nf);
        JcdotQdot = CalcContactJdotQdot(model, q, qdot, flag_contact, nf);


        M = Jc * model.Hinv * Jc';
        d = Jc * model.Hinv * (tau - model.C) + JcdotQdot;

        
        fqp = -M^-1 * d;

        % Check whether the Fz is positive
        [flag_contact, flag_recalc] = CheckContactForce(model, flag_contact, fqp, nf);
        if flag_recalc==0
            flcp = Jc' * fqp;
        end
    end % end of while

    % Calculate contact force from PD controller
    % fpd = CalcContactForcePD(q, qdot, flag_contact);

    % Get contact force for plot
    % [ip.fc, ip.fcqp, ip.fcpd] = GetContactForce(fqp, fpd, flag_contact);   

end

function [flag_newcontact, flag_recalc] = CheckContactForce(model, flag_contact, fqp, nf)    


    flag_recalc = 0;
    flag_newcontact = flag_contact;

    k = 0;
    for i=1:model.NC
        if flag_contact(i)~=0
            if fqp(k*nf+nf, 1) <0
                flag_newcontact(i) = 0;
                flag_recalc = 1;
                break;
            end

            k = k+1;
        end
    end

end