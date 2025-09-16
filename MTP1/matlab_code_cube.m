function run_outlier_analysis()
    % This is the main function that runs the entire analysis.
    % It calculates the outlier classifications and generates all plots and data files.

    % ===== MAIN SCRIPT =====
    % --- SETTINGS ---
    step = 0.025;
    % Prompt the user for a folder to save the output files
    folder_prompt = 'Enter folder to save all images and data (KL version): ';
    save_folder = input(folder_prompt, 's');
    if ~isfolder(save_folder)
        mkdir(save_folder);
    end

    % --- DATA GENERATION AND CLASSIFICATION ---
    all_points = {}; % Use a cell array to collect data points
    fprintf('Running outlier detection with KL Divergence...\n');
    t_values = 0:step:1;
    
    point_idx = 1;
    for t1 = t_values
        for t2 = t_values
            for t3 = t_values
                point_data.t1 = t1;
                point_data.t2 = t2;
                point_data.t3 = t3;
                
                if abs(t1 - t2) < 1e-9 && abs(t2 - t3) < 1e-9
                    point_data.outlier_class = "No Outlier";
                else
                    t = [t1, t2, t3];
                    % Represent as Bernoulli distributions
                    P1 = [t(1), 1-t(1)]; P2 = [t(2), 1-t(2)]; P3 = [t(3), 1-t(3)];
                    % Create leave-one-out reference distributions
                    Q1 = (P2 + P3) / 2; Q2 = (P1 + P3) / 2; Q3 = (P1 + P2) / 2;
                    % Calculate divergence scores
                    D = [kl_divergence(P1, Q1), kl_divergence(P2, Q2), kl_divergence(P3, Q3)];
                    
                    % --- Ambiguity Detection Logic ---
                    max_score = max(D);
                    indices_of_max = find(abs(D - max_score) < 1e-9);
                    
                    if length(indices_of_max) > 1
                        point_data.outlier_class = "Ambiguous";
                    else
                        [~, outlier_idx] = max(D);
                        point_data.outlier_class = sprintf("Outlier %d", outlier_idx);
                    end
                end
                all_points{point_idx} = point_data;
                point_idx = point_idx + 1;
            end
        end
    end
    fprintf('Calculation complete.\n');

    % --- DATA HANDLING AND SAVING ---
    fprintf('Organizing data and saving to files...\n');
    % Convert the cell array of structs to a MATLAB table
    T = struct2table([all_points{:}]);

    % Save to Excel
    excel_path = fullfile(save_folder, 'outlier_classification_data_kl.xlsx');
    writetable(T, excel_path);
    fprintf('Data saved to %s\n', excel_path);

    % Save to .mat file (MATLAB's native format)
    mat_path = fullfile(save_folder, 'outlier_classification_data_kl.mat');
    save(mat_path, 'T');
    fprintf('Data also saved to %s\n', mat_path);

    % Extract coordinate arrays for plotting using logical indexing
    is_red = (T.outlier_class == "Outlier 1");
    is_green = (T.outlier_class == "Outlier 2");
    is_blue = (T.outlier_class == "Outlier 3");
    is_black = (T.outlier_class == "No Outlier");
    is_orange = (T.outlier_class == "Ambiguous");

    x_red = T.t1(is_red); y_red = T.t2(is_red); z_red = T.t3(is_red);
    x_green = T.t1(is_green); y_green = T.t2(is_green); z_green = T.t3(is_green);
    x_blue = T.t1(is_blue); y_blue = T.t2(is_blue); z_blue = T.t3(is_blue);
    x_black = T.t1(is_black); y_black = T.t2(is_black); z_black = T.t3(is_black);
    x_orange = T.t1(is_orange); y_orange = T.t2(is_orange); z_orange = T.t3(is_orange);

    % =========================================================================
    % vv ALL 4 PLOTS ARE GENERATED BELOW THIS LINE vv
    % =========================================================================

    % --- PLOT 1: MAIN-STYLE SCATTER PLOT ---
    fprintf('Generating Plot 1: Main-Style Scatter...\n');
    fig1 = figure('Position', [100, 100, 800, 700]);
    ax1 = axes(fig1);
    hold(ax1, 'on');
    scatter3(ax1, x_red, y_red, z_red, 30, 'red', 'x');
    scatter3(ax1, x_green, y_green, z_green, 30, 'green', 'o', 'filled');
    scatter3(ax1, x_blue, y_blue, z_blue, 30, 'blue', 's', 'filled');
    scatter3(ax1, x_black, y_black, z_black, 40, 'black', 'd', 'filled');
    scatter3(ax1, x_orange, y_orange, z_orange, 40, [1, 0.6, 0], 'p', 'filled'); % Orange color
    hold(ax1, 'off');
    title(ax1, 'Main-Style Scatter Plot (KL Divergence)');
    xlabel(ax1, 't1'); ylabel(ax1, 't2'); zlabel(ax1, 't3');
    legend(ax1, 'Outlier 1', 'Outlier 2', 'Outlier 3', 'No Outlier', 'Ambiguous', 'Location', 'northwest');
    view(ax1, -60, 25); grid(ax1, 'on'); axis(ax1, 'equal');
    saveas(fig1, fullfile(save_folder, '1_main_scatter_kl.png'));

    % --- PLOT 2: MAIN-STYLE GROUPED PLOT ---
    fprintf('Generating Plot 2: Main-Style Grouped Shapes...\n');
    fig2 = figure('Position', [150, 150, 900, 900]);
    ax2 = axes(fig2);
    hold(ax2, 'on');
    plot_solid_cluster_shapes(ax2, [x_red, y_red, z_red], 'red', step);
    plot_solid_cluster_shapes(ax2, [x_green, y_green, z_green], 'green', step);
    plot_solid_cluster_shapes(ax2, [x_blue, y_blue, z_blue], 'blue', step);
    plot_solid_cluster_shapes(ax2, [x_orange, y_orange, z_orange], [1, 0.6, 0], step);
    plot3(ax2, [0, 1], [0, 1], [0, 1], 'k--', 'LineWidth', 2.5);
    hold(ax2, 'off');
    title(ax2, 'Main-Style Solid Shapes (KL Divergence)');
    xlabel(ax2, 't1'); ylabel(ax2, 't2'); zlabel(ax2, 't3');
    view(ax2, -60, 25); grid(ax2, 'on'); axis(ax2, 'equal');
    xlim(ax2, [0, 1]); ylim(ax2, [0, 1]); zlim(ax2, [0, 1]);
    h = zeros(5, 1);
    h(1) = patch(NaN, NaN, 'red', 'FaceAlpha', 0.5, 'EdgeColor', 'k');
    h(2) = patch(NaN, NaN, 'green', 'FaceAlpha', 0.5, 'EdgeColor', 'k');
    h(3) = patch(NaN, NaN, 'blue', 'FaceAlpha', 0.5, 'EdgeColor', 'k');
    h(4) = patch(NaN, NaN, [1, 0.6, 0], 'FaceAlpha', 0.5, 'EdgeColor', 'k');
    h(5) = plot(NaN, NaN, 'k--', 'LineWidth', 2);
    legend(h, 'Outlier 1 Region', 'Outlier 2 Region', 'Outlier 3 Region', 'Ambiguous Region', 'No Outlier', 'Location', 'northwest');
    saveas(fig2, fullfile(save_folder, '2_main_grouped_kl.png'));
    
    % --- PLOT 3: TEXTBOOK-STYLE SCATTER PLOT ---
    fprintf('Generating Plot 3: Textbook-Style Scatter...\n');
    fig3 = figure('Position', [200, 200, 950, 850]); % Made bigger
    ax3 = axes(fig3);
    hold(ax3, 'on');
    scatter3(ax3, x_red, y_red, z_red, 30, 'red', 'x');
    scatter3(ax3, x_green, y_green, z_green, 30, 'green', 'o', 'filled');
    scatter3(ax3, x_blue, y_blue, z_blue, 30, 'blue', 's', 'filled');
    scatter3(ax3, x_black, y_black, z_black, 40, 'black', 'd', 'filled');
    scatter3(ax3, x_orange, y_orange, z_orange, 40, [1, 0.6, 0], 'p', 'filled');
    draw_textbook_axes(ax3);
    hold(ax3, 'off');
    title(ax3, 'Textbook-Style Scatter Plot (KL Divergence)');
    legend(ax3, 'Outlier 1', 'Outlier 2', 'Outlier 3', 'No Outlier', 'Ambiguous', 'Location', 'northwest');
    view(ax3, -60, 25); axis(ax3, 'equal');
    saveas(fig3, fullfile(save_folder, '3_textbook_scatter_kl.png'));

    % --- PLOT 4: TEXTBOOK-STYLE GROUPED PLOT ---
    fprintf('Generating Plot 4: Textbook-Style Grouped Shapes...\n');
    fig4 = figure('Position', [250, 250, 1000, 1000]); % Made bigger
    ax4 = axes(fig4);
    hold(ax4, 'on');
    plot_solid_cluster_shapes(ax4, [x_red, y_red, z_red], 'red', step);
    plot_solid_cluster_shapes(ax4, [x_green, y_green, z_green], 'green', step);
    plot_solid_cluster_shapes(ax4, [x_blue, y_blue, z_blue], 'blue', step);
    plot_solid_cluster_shapes(ax4, [x_orange, y_orange, z_orange], [1, 0.6, 0], step);
    plot3(ax4, [0, 1], [0, 1], [0, 1], 'k--', 'LineWidth', 2.5);
    draw_textbook_axes(ax4);
    hold(ax4, 'off');
    title(ax4, 'Textbook-Style Solid Shapes (KL Divergence)');
    view(ax4, -60, 25); axis(ax4, 'equal');
    legend(h, 'Outlier 1 Region', 'Outlier 2 Region', 'Outlier 3 Region', 'Ambiguous Region', 'No Outlier', 'Location', 'northwest');
    saveas(fig4, fullfile(save_folder, '4_textbook_grouped_kl.png'));

    fprintf('\nProcess complete! All files saved in folder: ''%s''\n', save_folder);
end

% ===== LOCAL HELPER FUNCTIONS =====

function score = kl_divergence(p, q)
    % Calculates KL Divergence D_KL(p || q)
    epsilon = 1e-10;
    p_safe = p + epsilon;
    q_safe = q + epsilon;
    score = sum(p_safe .* log(p_safe ./ q_safe));
end

function plot_solid_cluster_shapes(ax, points, face_color, step)
    % Finds clusters and plots their convex hull as a solid shape.
    if isempty(points)
        return;
    end
    
    % DBSCAN to find clusters. eps is crucial and related to grid step size.
    labels = dbscan(points, step * 3.5, 20);
    unique_labels = unique(labels);
    unique_labels(unique_labels == -1) = []; % Remove noise points

    for k = 1:length(unique_labels)
        label = unique_labels(k);
        cluster_points = points(labels == label, :);
        if size(cluster_points, 1) < 4
            continue;
        end
        try
            % convhull returns the indices of points forming the hull faces
            hull_indices = convhull(cluster_points);
            % trisurf plots the triangular surface
            trisurf(hull_indices, cluster_points(:,1), cluster_points(:,2), cluster_points(:,3), ...
                'Parent', ax, 'FaceColor', face_color, 'FaceAlpha', 0.5, 'EdgeColor', 'k', 'LineWidth', 0.2);
        catch ME
            fprintf('Could not compute hull for a cluster: %s\n', ME.message);
        end
    end
end

function draw_textbook_axes(ax)
    % Draws clean, minimalist axes and numerical labels.
    axis(ax, 'off');
    axis_len = 1.1;
    % Draw axis arrows using quiver3
    quiver3(ax, 0,0,0, axis_len,0,0, 'k', 'LineWidth', 1.5, 'MaxHeadSize', 0.3);
    quiver3(ax, 0,0,0, 0,axis_len,0, 'k', 'LineWidth', 1.5, 'MaxHeadSize', 0.3);
    quiver3(ax, 0,0,0, 0,0,axis_len, 'k', 'LineWidth', 1.5, 'MaxHeadSize', 0.3);
    % Draw axis labels
    text(ax, axis_len, 0, 0, ' t1', 'FontSize', 14);
    text(ax, 0, axis_len, 0, ' t2', 'FontSize', 14);
    text(ax, 0, 0, axis_len, ' t3', 'FontSize', 14);
    % Draw tick values
    tick_values = 0.2:0.2:1.0;
    for val = tick_values
        text(ax, val, -0.05, 0, sprintf('%.1f', val), 'Color', [0.5 0.5 0.5], 'HorizontalAlignment', 'center');
        text(ax, -0.05, val, 0, sprintf('%.1f', val), 'Color', [0.5 0.5 0.5], 'HorizontalAlignment', 'center');
        text(ax, -0.05, 0, val, sprintf('%.1f', val), 'Color', [0.5 0.5 0.5], 'HorizontalAlignment', 'center');
    end
    xlim(ax, [-0.1, axis_len]); ylim(ax, [-0.1, axis_len]); zlim(ax, [-0.1, axis_len]);
end