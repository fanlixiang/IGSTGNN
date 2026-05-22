import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json

from src.base.model import BaseModel

class IGSTGNN(BaseModel):
    '''
    Incident-Guided Spatio-Temporal Graph Neural Network 
    '''
    def __init__(self, model_args, **args):
        # Extract BaseModel required parameters
        base_args = {k: v for k, v in args.items() if k in ['node_num', 'input_dim', 'output_dim', 'seq_len', 'horizon']}
        super(IGSTGNN, self).__init__(**base_args)
        self.model_args = model_args
        model_args.update(args)
        
        self._in_feat = model_args['num_feat'] 
        self._hidden_dim = model_args['num_hidden'] 
        self._node_dim = model_args['node_hidden'] 
        self._forecast_dim = 256 
        self._output_hidden = 512 

        self._node_num = self.node_num
        self._k_s = model_args['k_s'] 
        self._k_t = model_args['k_t'] 
        self._num_layers = model_args['layer']

        model_args.setdefault('use_pre', True)
        model_args.setdefault('dy_graph', True)
        model_args.setdefault('sta_graph', True)

        self._model_args = model_args 

        self.embedding = nn.Linear(self._in_feat, self._hidden_dim)

        self.node_emb_u = nn.Parameter(torch.empty(self._node_num, self._node_dim))
        self.node_emb_d = nn.Parameter(torch.empty(self._node_num, self._node_dim))
        self.T_i_D_emb = nn.Parameter(torch.empty(model_args['tpd'], model_args['time_emb_dim']))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, model_args['time_emb_dim']))

        self.layers = nn.ModuleList([DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim, **model_args)])
        for _ in range(self._num_layers - 1):
            self.layers.append(DecoupleLayer(self._hidden_dim, fk_dim=self._forecast_dim, **model_args))

        if model_args['dy_graph']:
            self.dynamic_graph_constructor = DynamicGraphConstructor(**model_args)

        self.out_fc_1 = nn.Linear(self._forecast_dim, self._output_hidden)
        self.out_fc_2 = nn.Linear(self._output_hidden, model_args['gap'])

        data_dir = model_args.get('data_path') or os.path.join('.', 'data', model_args['dataset'])
        
        with open(os.path.join(data_dir, 'desc_mapping.json'), 'r') as f:
            desc_mapping = json.load(f)
        with open(os.path.join(data_dir, 'type_mapping.json'), 'r') as f:
            type_mapping = json.load(f)
        
        self.icsf_module = IncidentContextSpatialFusion(
            hidden_dim=self._hidden_dim,
            time_emb_dim=model_args['time_emb_dim'],
            use_sensor_info=model_args.get('use_sensor_info', False),
            num_desc=len(desc_mapping),
            num_types=len(type_mapping),
            sensor_type_size=model_args.get('sensor_type_size', 1),
            surface_size=model_args.get('surface_size', 1),
            roadway_use_size=model_args.get('roadway_use_size', 1),
        )

        self.tiid_module = TemporalIncidentImpactDecay(
            incident_dim=self._hidden_dim,
            forecast_dim=self._forecast_dim,
            sigma_t=model_args.get('sigma_t', 1.0),
            incident_scale=model_args.get('lambda_incident', 1.0),
        )

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.node_emb_u)
        nn.init.xavier_uniform_(self.node_emb_d)
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)



    def _graph_constructor(self, **inputs):
        E_d = inputs['node_embedding_u']
        E_u = inputs['node_embedding_d']
        if self._model_args['sta_graph']:
            static_graph = [F.softmax(F.relu(torch.mm(E_d, E_u.T)), dim=1)]
        else:
            static_graph = []
        if self._model_args['dy_graph']:
            dynamic_graph = self.dynamic_graph_constructor(**inputs)
        else:
            dynamic_graph = []
        return static_graph, dynamic_graph

    def _prepare_inputs(self, history_data):
        num_feat = self._model_args['num_feat']
        node_emb_u = self.node_emb_u
        node_emb_d = self.node_emb_d
        time_index = (history_data[:, :, :, num_feat] * self._model_args['tpd']).long()
        day_index = (history_data[:, :, :, num_feat + 1] * 7).long()
        time_index = time_index.clamp(0, self._model_args['tpd'] - 1)
        day_index = day_index.clamp(0, 6)
        time_in_day_feat = self.T_i_D_emb[time_index]
        day_in_week_feat = self.D_i_W_emb[day_index]
        history_data = history_data[:, :, :, :num_feat]
        return history_data, node_emb_u, node_emb_d, time_in_day_feat, day_in_week_feat

    def forward(self, history_data, label=None, incident_data=None, sensor_data=None):  # (b, t, n, f)
        history_data, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat = self._prepare_inputs(history_data)

        history_data = self.embedding(history_data)
        incident_context = None
        if incident_data is not None:
            incident_tod_feat = time_in_day_feat[:, -1, 0, :]
            incident_day_feat = day_in_week_feat[:, -1, 0, :]
            history_data, incident_context = self.icsf_module(
                history_data,
                incident_data,
                sensor_data,
                incident_tod_feat,
                incident_day_feat,
            )
        static_graph, dynamic_graph = self._graph_constructor(node_embedding_u=node_embedding_u, node_embedding_d=node_embedding_d, 
                                                              history_data=history_data, time_in_day_feat=time_in_day_feat, 
                                                              day_in_week_feat=day_in_week_feat)

        dif_forecast_hidden_list = []
        inh_forecast_hidden_list = []

        inh_backcast_seq_res = history_data
        for _, layer in enumerate(self.layers):
            inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden = layer(inh_backcast_seq_res, 
                                                                                  dynamic_graph, static_graph, 
                                                                                  node_embedding_u, node_embedding_d, 
                                                                                  time_in_day_feat, day_in_week_feat)
            dif_forecast_hidden_list.append(dif_forecast_hidden)
            inh_forecast_hidden_list.append(inh_forecast_hidden)

        dif_forecast_hidden = sum(dif_forecast_hidden_list)
        inh_forecast_hidden = sum(inh_forecast_hidden_list)
        forecast_hidden = dif_forecast_hidden + inh_forecast_hidden

        if incident_context is not None:
            forecast_hidden = self.tiid_module(forecast_hidden, incident_context)

        forecast = self.out_fc_2(F.relu(self.out_fc_1(F.relu(forecast_hidden))))

        forecast = forecast.transpose(1,2).contiguous().view(forecast.shape[0], forecast.shape[2], -1)
        return forecast.transpose(1, 2).unsqueeze(-1)


class TemporalIncidentImpactDecay(nn.Module):
    """
    Temporal Incident Impact Decay (TIID).

    The module follows the paper pipeline: the ST backbone first produces
    hidden states over the prediction horizon, then TIID adds a temporally
    decayed incident context before the prediction head maps hidden states
    to traffic forecasts.
    """
    def __init__(self, incident_dim, forecast_dim, sigma_t=1.0, incident_scale=1.0):
        super().__init__()
        self.context_projection = nn.Linear(incident_dim, forecast_dim, bias=False)
        self.register_buffer('sigma_t', torch.tensor(float(sigma_t)))
        self.incident_scale = float(incident_scale)

    def forward(self, forecast_hidden, incident_context):
        forecast_len = forecast_hidden.shape[1]
        time_steps = torch.arange(
            1,
            forecast_len + 1,
            dtype=forecast_hidden.dtype,
            device=forecast_hidden.device,
        )
        sigma_t = self.sigma_t.to(device=forecast_hidden.device, dtype=forecast_hidden.dtype)
        temporal_decay = torch.exp(-(time_steps ** 2) / (2 * sigma_t ** 2))
        temporal_decay = temporal_decay.view(1, forecast_len, 1, 1)

        projected_context = self.context_projection(incident_context)
        temporal_context = projected_context.unsqueeze(1) * temporal_decay
        return forecast_hidden + self.incident_scale * temporal_context


class IncidentContextSpatialFusion(nn.Module):
    """
    Incident-Context Spatial Fusion (ICSF).
    """
    def __init__(
        self,
        hidden_dim,
        time_emb_dim,
        use_sensor_info,
        num_desc,
        num_types,
        sensor_type_size=1,
        surface_size=1,
        roadway_use_size=1,
    ):
        super(IncidentContextSpatialFusion, self).__init__()

        self.position_emb_dim = 8
        self.desc_emb_dim = 32
        self.type_emb_dim = 8
        self.holiday_emb_dim = 4
        self.context_time_emb_dim = 2 * time_emb_dim

        self.position_embedding = nn.Embedding(12, self.position_emb_dim)
        self.desc_embedding = nn.Embedding(num_desc, self.desc_emb_dim)
        self.incident_type_embedding = nn.Embedding(num_types, self.type_emb_dim)
        self.holiday_embedding = nn.Embedding(2, self.holiday_emb_dim)

        self.incident_input_dim = (
            self.position_emb_dim
            + self.desc_emb_dim
            + self.type_emb_dim
            + self.holiday_emb_dim
            + self.context_time_emb_dim
        )
        self.incident_fusion = nn.Sequential(
            nn.Linear(self.incident_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )
        
        self.icsf_dim = hidden_dim
        self.q_proj = nn.Linear(hidden_dim, self.icsf_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.icsf_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.icsf_dim, bias=False)
        self.distance_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.icsf_dim)
        )
        
        self.use_sensor_info = use_sensor_info
        
        if self.use_sensor_info:
            self.sensor_feature_emb_dim = 8
            self.sensor_type_embedding = nn.Embedding(max(sensor_type_size, 1), self.sensor_feature_emb_dim)
            self.surface_embedding = nn.Embedding(max(surface_size, 1), self.sensor_feature_emb_dim)
            self.roadway_use_embedding = nn.Embedding(max(roadway_use_size, 1), self.sensor_feature_emb_dim)
            
            self.road_width_linear = nn.Linear(1, self.sensor_feature_emb_dim)
            self.speed_limit_linear = nn.Linear(1, self.sensor_feature_emb_dim)
            
            sensor_embed_dim = self.sensor_feature_emb_dim * 5
            self.sensor_fusion = nn.Sequential(
                nn.Linear(sensor_embed_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, self.icsf_dim)
            )
        
        self.attn_dim = self.icsf_dim
        self.distance_dim = self.icsf_dim
        self.sensor_dim = self.icsf_dim if self.use_sensor_info else 0
        self.icsf_fusion_dim = self.attn_dim + self.distance_dim + self.sensor_dim
        
        self.icsf_fusion_mlp = nn.Sequential(
            nn.Linear(self.icsf_fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.icsf_dim)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def embed_incident_features(self, incident_data, incident_tod_feat=None, incident_day_feat=None):
        """
        Embed incident features
        """
        incident_features = incident_data['incident']
        batch_size = incident_features.size(0)
        embeddings = []
        
        position = incident_data['position'].long().clamp(0, self.position_embedding.num_embeddings - 1)
        position_emb = self.position_embedding(position)
        embeddings.append(position_emb)

        description = incident_features[:, 1].long().clamp(0, self.desc_embedding.num_embeddings - 1)
        desc_emb = self.desc_embedding(description)
        embeddings.append(desc_emb)

        incident_type = incident_features[:, 2].long().clamp(0, self.incident_type_embedding.num_embeddings - 1)
        type_emb = self.incident_type_embedding(incident_type)
        embeddings.append(type_emb)

        holiday = incident_features[:, 3].long().clamp(0, self.holiday_embedding.num_embeddings - 1)
        holiday_emb = self.holiday_embedding(holiday)
        embeddings.append(holiday_emb)

        if incident_tod_feat is None:
            incident_tod_feat = torch.zeros(
                batch_size,
                self.context_time_emb_dim // 2,
                device=incident_features.device,
                dtype=incident_features.dtype,
            )
        if incident_day_feat is None:
            incident_day_feat = torch.zeros(
                batch_size,
                self.context_time_emb_dim // 2,
                device=incident_features.device,
                dtype=incident_features.dtype,
            )
        embeddings.append(incident_tod_feat)
        embeddings.append(incident_day_feat)

        combined_embedding = torch.cat(embeddings, dim=1)
        incident_embedding = self.incident_fusion(combined_embedding)
        
        return incident_embedding

    def get_attention_weights(self, distances):
        """
        Calculate distance-based attention weights
        """
        attention_logits = self.distance_encoder(distances)
        attention_weights = F.softmax(attention_logits, dim=1)
            
        return attention_weights

    def process_sensor_info(self, sensor_data, num_nodes):
        """
        Process sensor information, return sensor embeddings
        """
        if not self.use_sensor_info or sensor_data is None:
            return None

        sensor_type = sensor_data['sensor_type'].long().clamp(0, self.sensor_type_embedding.num_embeddings - 1)
        surface = sensor_data['surface'].long().clamp(0, self.surface_embedding.num_embeddings - 1)
        roadway_use = sensor_data['roadway_use'].long().clamp(0, self.roadway_use_embedding.num_embeddings - 1)

        sensor_type_emb = self.sensor_type_embedding(sensor_type)
        surface_emb = self.surface_embedding(surface)
        roadway_use_emb = self.roadway_use_embedding(roadway_use)
        
        road_width_emb = self.road_width_linear(sensor_data['road_width'].float().unsqueeze(-1))
        speed_limit_emb = self.speed_limit_linear(sensor_data['speed_limit'].float().unsqueeze(-1))
        
        sensor_features = torch.cat([
            sensor_type_emb,
            surface_emb,
            roadway_use_emb,
            road_width_emb,
            speed_limit_emb
        ], dim=-1)
        
        sensor_embed = self.sensor_fusion(sensor_features)
        
        return sensor_embed

    def forward(self, history_data, incident_data, sensor_data=None, incident_tod_feat=None, incident_day_feat=None):
        """
        Fuse the current incident context into the latest traffic hidden state.
        """
        _, _, num_nodes, hidden_dim = history_data.shape
        
        incident_embedding = self.embed_incident_features(incident_data, incident_tod_feat, incident_day_feat)
        
        incident_distances = incident_data['distances']
        
        distance_mask = (incident_distances.abs().sum(dim=-1) > 0).float().unsqueeze(-1)
        
        distance_context = self.get_attention_weights(incident_distances)
        
        K_incident = self.k_proj(incident_embedding).unsqueeze(1)
        V_incident = self.v_proj(incident_embedding).unsqueeze(1)
        
        K_expanded = K_incident.expand(-1, num_nodes, -1)
        V_expanded = V_incident.expand(-1, num_nodes, -1)

        Q = self.q_proj(history_data[:, -1, :, :])
        attn_logits = (Q * K_expanded).sum(dim=-1, keepdim=True) / math.sqrt(hidden_dim)
        attn_logits_masked = attn_logits.masked_fill(distance_mask == 0, -1e7)
        semantic_attention = F.softmax(attn_logits_masked, dim=1) * distance_mask

        fusion_parts = [semantic_attention.expand(-1, -1, hidden_dim), distance_context]
        if self.use_sensor_info:
            sensor_embed = self.process_sensor_info(sensor_data, num_nodes)
            if sensor_embed is None:
                sensor_embed = torch.zeros_like(distance_context)
            fusion_parts.append(sensor_embed)

        fusion_input = torch.cat(fusion_parts, dim=-1)
            
        enhanced_attn = self.icsf_fusion_mlp(fusion_input)
        enhanced_attn_masked = enhanced_attn.masked_fill(distance_mask == 0, -1e7)
        final_attn_weights = F.softmax(enhanced_attn_masked, dim=1) * distance_mask
            
        incident_context = final_attn_weights * V_expanded
        enhanced_data = history_data.clone()
        enhanced_data[:, -1, :, :] = self.output_norm(history_data[:, -1, :, :] + incident_context)
        
        return enhanced_data, incident_context
        

class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, **model_args):
        super().__init__()
        self.estimation_gate = EstimationGate(node_emb_dim=model_args['node_hidden'], time_emb_dim=model_args['time_emb_dim'], hidden_dim=64)
        self.dif_layer = DifBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)
        self.inh_layer = InhBlock(hidden_dim, forecast_hidden_dim=fk_dim, **model_args)


    def forward(self, history_data, dynamic_graph, static_graph, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat):
        gated_history_data = self.estimation_gate(node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat, history_data)

        dif_backcast_seq_res, dif_forecast_hidden = self.dif_layer(history_data=history_data, gated_history_data=gated_history_data, dynamic_graph=dynamic_graph, static_graph=static_graph)

        inh_backcast_seq_res, inh_forecast_hidden = self.inh_layer(dif_backcast_seq_res)         
        return inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden


class EstimationGate(nn.Module):
    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim):
        super().__init__()
        self.fully_connected_layer_1 = nn.Linear(2 * node_emb_dim + time_emb_dim * 2, hidden_dim)
        self.activation = nn.ReLU()
        self.fully_connected_layer_2 = nn.Linear(hidden_dim, 1)


    def forward(self, node_embedding_u, node_embedding_d, time_in_day_feat, day_in_week_feat, history_data):
        batch_size, seq_length, _, _ = time_in_day_feat.shape
        estimation_gate_feat = torch.cat([time_in_day_feat, day_in_week_feat, node_embedding_u.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length,  -1, -1), node_embedding_d.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length,  -1, -1)], dim=-1)
        hidden = self.fully_connected_layer_1(estimation_gate_feat)
        hidden = self.activation(hidden)

        estimation_gate = torch.sigmoid(self.fully_connected_layer_2(hidden))[:, -history_data.shape[1]:, :, :]
        history_data = history_data * estimation_gate
        return history_data


class ResidualDecomp(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        self.ac = nn.ReLU()


    def forward(self, x, y):
        u = x - self.ac(y)
        u = self.ln(u)
        return u


class DifBlock(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=256, **model_args):
        super().__init__()
        self.pre_defined_graph = model_args['adjs']

        self.localized_st_conv = STLocalizedConv(hidden_dim, pre_defined_graph=self.pre_defined_graph, **model_args)

        self.forecast_branch = DifForecast(hidden_dim, forecast_hidden_dim=forecast_hidden_dim, **model_args)
        self.backcast_branch = nn.Linear(hidden_dim, hidden_dim)
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])


    def forward(self, history_data, gated_history_data, dynamic_graph, static_graph):
        hidden_states_dif = self.localized_st_conv(gated_history_data, dynamic_graph, static_graph)

        forecast_hidden = self.forecast_branch(gated_history_data, hidden_states_dif, self.localized_st_conv, dynamic_graph, static_graph)
        backcast_seq = self.backcast_branch(hidden_states_dif)

        history_data = history_data[:, -backcast_seq.shape[1]:, :, :]
        backcast_seq_res = self.residual_decompose(history_data, backcast_seq)

        return backcast_seq_res, forecast_hidden



class STLocalizedConv(nn.Module):
    def __init__(self, hidden_dim, pre_defined_graph=None, **model_args):
        super().__init__()
        self.k_s = model_args['k_s']
        self.k_t = model_args['k_t']
        self.hidden_dim = hidden_dim

        self.pre_defined_graph = pre_defined_graph or model_args.get('adjs', [])
        self.use_predefined_graph = model_args.get('use_pre', False)
        self.use_dynamic_hidden_graph = model_args.get('dy_graph', True)
        self.use_static_hidden_graph = model_args.get('sta_graph', True)

        self.support_len = len(self.pre_defined_graph) + \
            int(self.use_dynamic_hidden_graph) + int(self.use_static_hidden_graph)
        self.num_matric = (int(self.use_predefined_graph) * len(self.pre_defined_graph) + len(
            self.pre_defined_graph) * int(self.use_dynamic_hidden_graph) + int(self.use_static_hidden_graph)) * self.k_s + 1
        self.dropout = nn.Dropout(model_args['dropout'])
        self.pre_defined_graph = self.get_graph(self.pre_defined_graph)

        self.fc_list_updt = nn.Linear(
            self.k_t * hidden_dim, self.k_t * hidden_dim, bias=False)
        self.gcn_updt = nn.Linear(
            self.hidden_dim * self.num_matric, self.hidden_dim)

        self.bn = nn.BatchNorm2d(self.hidden_dim)
        self.activation = nn.ReLU()


    def gconv(self, support, X_k, X_0):
        out = [X_0]
        for graph in support:
            if len(graph.shape) == 2:
                pass
            else:
                graph = graph.unsqueeze(1)
            H_k = torch.matmul(graph, X_k)
            out.append(H_k)
        out = torch.cat(out, dim=-1)
        out = self.gcn_updt(out)
        out = self.dropout(out)
        return out


    def get_graph(self, support):
        if not support:
            return []
        graph_ordered = []
        mask = 1 - torch.eye(support[0].shape[0]).to(support[0].device)
        for graph in support:
            k_1_order = graph
            graph_ordered.append(k_1_order * mask)

            for k in range(2, self.k_s + 1):
                k_1_order = torch.matmul(graph, k_1_order)
                graph_ordered.append(k_1_order * mask)

        st_local_graph = []
        for graph in graph_ordered:
            graph = graph.unsqueeze(-2).expand(-1, self.k_t, -1)
            graph = graph.reshape(
                graph.shape[0], graph.shape[1] * graph.shape[2])
            st_local_graph.append(graph)

        return st_local_graph


    def forward(self, X, dynamic_graph, static_graph):
        X = X.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3)
        batch_size, seq_len, node_num, kernel_size, num_feat = X.shape

        support = []
        if self.use_predefined_graph:
            support = support + self.pre_defined_graph
        if self.use_dynamic_hidden_graph:
            support = support + dynamic_graph
        if self.use_static_hidden_graph:
            support = support + self.get_graph(static_graph)

        # parallelize
        X = X.reshape(batch_size, seq_len, node_num, kernel_size * num_feat)
        out = self.fc_list_updt(X)
        out = self.activation(out)
        out = out.view(batch_size, seq_len, node_num, kernel_size, num_feat)
        X_0 = torch.mean(out, dim=-2)
        X_k = out.transpose(-3, -2).reshape(batch_size, seq_len, kernel_size*node_num, num_feat)
        hidden = self.gconv(support, X_k, X_0)
        return hidden


class DifForecast(nn.Module):
    def __init__(self, hidden_dim, forecast_hidden_dim=None, **model_args):
        super().__init__()
        self.k_t = model_args['k_t']
        self.output_seq_len = model_args['seq_len']
        self.forecast_fc = nn.Linear(hidden_dim, forecast_hidden_dim)
        self.model_args = model_args


    def forward(self, gated_history_data, hidden_states_dif, localized_st_conv, dynamic_graph, static_graph):
        predict = []
        history = gated_history_data
        predict.append(hidden_states_dif[:, -1, :, :].unsqueeze(1))
        for _ in range(int(self.output_seq_len / self.model_args['gap'])-1):
            _1 = predict[-self.k_t:]
            if len(_1) < self.k_t:
                sub = self.k_t - len(_1)
                _2 = history[:, -sub:, :, :]
                _1 = torch.cat([_2] + _1, dim=1)
            else:
                _1 = torch.cat(_1, dim=1)
            predict.append(localized_st_conv(_1, dynamic_graph, static_graph))
        predict = torch.cat(predict, dim=1)
        predict = self.forecast_fc(predict)
        return predict


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=None, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)


    def forward(self, X):
        X = X + self.pe[:X.size(0)]
        X = self.dropout(X)
        return X


class InhBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, bias=True, forecast_hidden_dim=256, **model_args):
        super().__init__()
        self.num_feat = hidden_dim
        self.hidden_dim = hidden_dim

        self.pos_encoder = PositionalEncoding(hidden_dim, model_args['dropout'])
        self.rnn_layer = RNNLayer(hidden_dim, model_args['dropout'])
        self.transformer_layer = TransformerLayer(hidden_dim, num_heads, model_args['dropout'], bias)

        self.forecast_block = InhForecast(hidden_dim, forecast_hidden_dim, **model_args)
        self.backcast_fc = nn.Linear(hidden_dim, hidden_dim)
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])


    def forward(self, hidden_inherent_signal):
        [batch_size, seq_len, node_num, num_feat] = hidden_inherent_signal.shape

        hidden_states_rnn = self.rnn_layer(hidden_inherent_signal)
        hidden_states_rnn = self.pos_encoder(hidden_states_rnn)                   
        hidden_states_inh = self.transformer_layer(hidden_states_rnn, hidden_states_rnn, hidden_states_rnn)

        forecast_hidden = self.forecast_block(hidden_inherent_signal, hidden_states_rnn, hidden_states_inh, self.transformer_layer, self.rnn_layer, self.pos_encoder)
        hidden_states_inh = hidden_states_inh.reshape(seq_len, batch_size, node_num, num_feat)
        hidden_states_inh = hidden_states_inh.transpose(0, 1)
        backcast_seq = self.backcast_fc(hidden_states_inh)                                   
        backcast_seq_res= self.residual_decompose(hidden_inherent_signal, backcast_seq)
        return backcast_seq_res, forecast_hidden


class RNNLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, X):
        [batch_size, seq_len, node_num, hidden_dim] = X.shape
        X = X.transpose(1, 2).reshape(batch_size * node_num, seq_len, hidden_dim)
        hx = torch.zeros_like(X[:, 0, :])
        output  = []
        for _ in range(X.shape[1]):
            hx = self.gru_cell(X[:, _, :], hx)
            output.append(hx)
        output = torch.stack(output, dim=0)
        output = self.dropout(output)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.multi_head_self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)


    def forward(self, X, K, V):
        hidden_states_MSA = self.multi_head_self_attention(X, K, V)[0]
        hidden_states_MSA = self.dropout(hidden_states_MSA)
        return hidden_states_MSA


class InhForecast(nn.Module):
    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.output_seq_len = model_args['seq_len']
        self.model_args = model_args

        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)


    def forward(self, X, RNN_H, Z, transformer_layer, rnn_layer, pe):
        [batch_size, _, node_num, num_feat] = X.shape

        predict = [Z[-1, :, :].unsqueeze(0)]
        for _ in range(int(self.output_seq_len / self.model_args['gap'])-1):
            _gru = rnn_layer.gru_cell(predict[-1][0], RNN_H[-1]).unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _gru], dim=0)

            if pe is not None:
                RNN_H = pe(RNN_H)

            _Z = transformer_layer(_gru, K=RNN_H, V=RNN_H)
            predict.append(_Z)
        
        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(-1, batch_size, node_num, num_feat)
        predict = predict.transpose(0, 1)
        predict = self.forecast_fc(predict)
        return predict


class DynamicGraphConstructor(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.k_s = model_args['k_s']
        self.k_t = model_args['k_t']
        self.hidden_dim = model_args['num_hidden']
        self.node_dim = model_args['node_hidden']

        self.distance_function = DistanceFunction(**model_args)
        self.mask = Mask(**model_args)
        self.normalizer = Normalizer()
        self.multi_order = MultiOrder(order=self.k_s)


    def st_localization(self, graph_ordered):
        st_local_graph = []
        for modality_i in graph_ordered:
            for k_order_graph in modality_i:
                k_order_graph = k_order_graph.unsqueeze(
                    -2).expand(-1, -1, self.k_t, -1)
                k_order_graph = k_order_graph.reshape(
                    k_order_graph.shape[0], k_order_graph.shape[1], k_order_graph.shape[2] * k_order_graph.shape[3])
                st_local_graph.append(k_order_graph)
        return st_local_graph


    def forward(self, **inputs):
        X = inputs['history_data']
        E_d = inputs['node_embedding_d']
        E_u = inputs['node_embedding_u']
        T_D = inputs['time_in_day_feat']
        D_W = inputs['day_in_week_feat']
        
        dist_mx = self.distance_function(X, E_d, E_u, T_D, D_W)
        dist_mx = self.mask(dist_mx)
        dist_mx = self.normalizer(dist_mx)
        mul_mx = self.multi_order(dist_mx)
        dynamic_graphs = self.st_localization(mul_mx)
        return dynamic_graphs


class DistanceFunction(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.hidden_dim = model_args['num_hidden']
        self.node_dim = model_args['node_hidden']
        self.time_slot_emb_dim = self.hidden_dim
        self.input_seq_len = model_args['seq_len']
        self.dropout = nn.Dropout(model_args['dropout'])
        self.fc_ts_emb1 = nn.Linear(self.input_seq_len, self.hidden_dim * 2)
        self.fc_ts_emb2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.time_slot_embedding = nn.Linear(model_args['time_emb_dim'], self.time_slot_emb_dim)

        self.ts_feat_dim = self.hidden_dim
        self.all_feat_dim = self.ts_feat_dim + self.node_dim + model_args['time_emb_dim'] * 2

        self.WQ = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.WK = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.bn = nn.BatchNorm1d(self.hidden_dim * 2)


    def reset_parameters(self):
        for q_vec in self.q_vecs:
            nn.init.xavier_normal_(q_vec.data)
        for bias in self.biases:
            nn.init.zeros_(bias.data)


    def forward(self, X, E_d, E_u, T_D, D_W):
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]

        X = X[:, :, :, 0].transpose(1, 2).contiguous()
        [batch_size, node_num, seq_len] = X.shape
        X = X.view(batch_size * node_num, seq_len)
        dy_feat = self.fc_ts_emb2(self.dropout(self.bn(F.relu(self.fc_ts_emb1(X)))))
        dy_feat = dy_feat.view(batch_size, node_num, -1)

        emb1 = E_d.unsqueeze(0).expand(batch_size, -1, -1)
        emb2 = E_u.unsqueeze(0).expand(batch_size, -1, -1)

        X1 = torch.cat([dy_feat, T_D, D_W, emb1], dim=-1)
        X2 = torch.cat([dy_feat, T_D, D_W, emb2], dim=-1)

        X = [X1, X2]
        adjacent_list = []
        for _ in X:
            Q = self.WQ(_)
            K = self.WK(_)
            QKT = torch.bmm(Q, K.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
            W = torch.softmax(QKT, dim=-1)
            adjacent_list.append(W)
        return adjacent_list


class Mask(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.mask = model_args['adjs']


    def _mask(self, index, adj):
        mask = self.mask[index] + torch.ones_like(self.mask[index]) * 1e-7
        return mask.to(adj.device) * adj


    def forward(self, adj):
        result = []
        for index, _ in enumerate(adj):
            result.append(self._mask(index, _))
        return result


class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()


    def _remove_nan_inf(self, tensor):
        tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
        tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
        return tensor


    def _norm(self, graph):
        degree = torch.sum(graph, dim=2)
        degree = self._remove_nan_inf(1 / degree)
        degree = torch.diag_embed(degree)
        normed_graph = torch.bmm(degree, graph)
        return normed_graph


    def forward(self, adj):
        return [self._norm(_) for _ in adj]


class MultiOrder(nn.Module):
    def __init__(self, order=2):
        super().__init__()
        self.order = order


    def _multi_order(self, graph):
        graph_ordered = []
        k_1_order = graph
        mask = torch.eye(graph.shape[1]).to(graph.device)
        mask = 1 - mask
        graph_ordered.append(k_1_order * mask)
        for k in range(2, self.order + 1):
            k_1_order = torch.matmul(k_1_order, graph)
            graph_ordered.append(k_1_order * mask)
        return graph_ordered


    def forward(self, adj):
        return [self._multi_order(_) for _ in adj]
