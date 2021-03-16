function fpd = CalcContactForcePD(model, q, qdot, flag_contact, contact_force_kp, contact_force_kd, nf)
%CalcContactForcePD - Calculate contact force by PD controller based on contact point pos and vel 
%
% Syntax: fpd = CalcContactForcePD(q, qdot, flag_contact)
%
% Long description

    if sum(flag_contact)==0
        fpd = zeros(model.NC*nf, 1);
    else
        endpos = zeros(3, model.NC);
        endvel = zeros(3, model.NC);
        k = 0;
        for i=1:model.NC
            if flag_contact(i) ~= 0
                % Calcualte pos and vel of foot endpoint
                endpos(:, i) = CalcBodyToBaseCoordinates(model, q, model.idcontact(i), model.contactpoint{i});
                endvel(:, i) = CalcPointVelocity(model, q, qdot, model.idcontact(i), model.contactpoint{i});
                
                % Calculate contact force by PD controller
                if nf==2
                    fpdi(1, 1) = -contact_force_kp(1)*endvel(1, i);
                    fpdi(2, 1) = -contact_force_kp(3)*endpos(3, i) - contact_force_kd(3) * min(endvel(3, i), 0.0);
                else
                    if nf==3
                        fpdi(1, 1) = -contact_force_kp(1)*endvel(1, i);
                        fpdi(2, 1) = -contact_force_kp(2)*endvel(2, i);
                        fpdi(3, 1) = -contact_force_kp(3)*min(endpos(3, i), 0.0) - contact_force_kd(3) * min(endvel(3, i), 0.0);
                    end
                end

                fpd(k*nf+1:k*nf+nf, 1) = fpdi;  
                
                k = k+1;
            end
        end % end of for
    end
end