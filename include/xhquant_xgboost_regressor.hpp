////////////////////////////////////////////////////////////////////////
/// \file      xhquant_xgboost_regressor.hpp
/// \copyright Copyright (c) 2022 标准模型投资有限公司.
///            All rights reserved.
///            Licensed under the MIT License.
/// \author    金小海
/// \date      2022年08月20日, Sat, 09:14
/// \version   1.0
/// \brief
#ifndef XHQUANT_XGBOOST_REGRESSOR_HPP
#define XHQUANT_XGBOOST_REGRESSOR_HPP

#include <vector>
#include <string>
#include <sstream>
#include <map>


namespace xhquant
{
    namespace model
    {
        class xhquant_xgboost_regressor
        {
            template<class data_type>
            struct numeric_after_substr_output
            {
                data_type value = 0;
                bool found = false;
                bool failed = true;
                std::string rest;
            };

        public:
            explicit xhquant_xgboost_regressor(std::vector<std::string> _features, float _base_score = 0.5);

            virtual ~xhquant_xgboost_regressor();

        public:
            void init(std::string &model);

            float forward(const float *input) const;

        private:
            static bool is_integer(const std::string &s);

            static std::vector<std::string> split(std::string const &str, const std::string &delimiter);

            void terminate_tree(int &num_previous_nodes, int &num_previous_leaves, std::map<int, int> &node_indices, std::map<int, int> &leaf_indices, int &trees_skipped);

            static void correct_indices(std::vector<int>::iterator begin, std::vector<int>::iterator end, std::map<int, int> const &node_indices, std::map<int, int> const &leaf_indices);

            template<class data_type>
            numeric_after_substr_output<data_type> numeric_after_substr(std::string const &str, std::string const &substr);

        public:
            std::vector<int> tree_root_index;

            std::vector<unsigned int> tree_cut_indices;
            std::vector<float> tree_cut_values;

            std::vector<int> tree_left_indices;
            std::vector<int> tree_right_indices;

            std::vector<int> tree_numbers;
            std::vector<float> tree_responses;
            std::vector<float> tree_base_responses;

            // 特征
            std::vector<std::string> features;
            float base_score;
        };


        ////////////////////////////////////////////////////////////////////////
        /// \brief
        /// \param _features
        /// \param _base_score
        inline
        xhquant_xgboost_regressor::xhquant_xgboost_regressor(std::vector<std::string> _features, float _base_score)
                : features(std::move(_features)), base_score(_base_score)
        {}


        ////////////////////////////////////////////////////////////////////////
        /// \brief
        inline
        xhquant_xgboost_regressor::~xhquant_xgboost_regressor()
        {
            tree_root_index.clear();
            tree_cut_indices.clear();
            tree_cut_values.clear();
            tree_left_indices.clear();
            tree_right_indices.clear();
            tree_responses.clear();
            tree_numbers.clear();
            tree_base_responses.clear();
            features.clear();
        }


        ////////////////////////////////////////////////////////////////////////
        /// \brief
        /// \param model
        inline
        void xhquant_xgboost_regressor::init(std::string &model)
        {
            int num_classes = features.size();
            tree_base_responses.resize(num_classes == 2 ? 1 : num_classes);

            int trees_skipped = 0;
            int num_variables = 0;
            std::map<std::string, int> variables_indices;
            bool fix_features = false;

            if (!features.empty())
            {
                fix_features = true;
                num_variables = features.size();
                for (int i = 0; i < num_variables; ++i)
                {
                    variables_indices[features[i]] = i;
                }
            }

            std::map<int, int> node_indices;
            std::map<int, int> leaf_indices;

            int num_previous_nodes = 0;
            int num_previous_leaves = 0;
            std::vector<std::string> model_lines = split(model, "\n");
            for (const auto &line: model_lines)
            {
                std::size_t start_position = line.find('[');
                std::size_t end_position = line.find(']');
                if (start_position != std::string::npos)  // 非叶子处理
                {
                    std::string sub_line = line.substr(start_position + 1, end_position - start_position - 1);
                    if (is_integer(sub_line) && !tree_responses.empty())
                    {
                        terminate_tree(num_previous_nodes, num_previous_leaves, node_indices, leaf_indices, trees_skipped);
                    }
                    else if (!is_integer(sub_line))
                    {
                        std::stringstream ss(line);
                        int index;         // 索引
                        ss >> index;

                        std::vector<std::string> split_string = split(sub_line, "<");
                        std::string const &feature_name = split_string[0];               // 分裂特征名字
                        float feature_cut_value = std::stof(split_string[1]);            // 分裂特征值

                        if (!variables_indices.count(feature_name))
                        {
                            if (fix_features)
                            {
                                throw std::runtime_error("feature " + feature_name + " not in list of features");
                            }
                            variables_indices[feature_name] = num_variables;
                            features.push_back(feature_name);
                            ++num_variables;
                        }

                        int yes, no;
                        numeric_after_substr_output<int> output = numeric_after_substr<int>(line, "yes=");
                        if (!output.failed)
                        {
                            yes = output.value;
                        }
                        else
                        {
                            throw std::runtime_error("problem while parsing the text dump");
                        }

                        output = numeric_after_substr<int>(output.rest, "no=");
                        if (!output.failed)
                        {
                            no = output.value;
                        }
                        else
                        {
                            throw std::runtime_error("problem while parsing the text dump");
                        }

                        tree_cut_values.push_back(feature_cut_value);
                        tree_cut_indices.push_back(variables_indices[feature_name]);
                        tree_left_indices.push_back(yes);
                        tree_right_indices.push_back(no);
                        node_indices[index] = node_indices.size() + num_previous_nodes;
                    }
                }
                else       // 处理叶子
                {
                    numeric_after_substr_output<float> output = numeric_after_substr<float>(line, "leaf=");
                    if (output.found)
                    {
                        std::stringstream ss(line);
                        int index;
                        ss >> index;

                        tree_responses.push_back(output.value);
                        leaf_indices[index] = leaf_indices.size() + num_previous_leaves;
                    }
                }
            }

            terminate_tree(num_previous_nodes, num_previous_leaves, node_indices, leaf_indices, trees_skipped);

            if (num_classes > 2 && (tree_root_index.size() + trees_skipped) % num_classes != 0)
            {
                std::cout << "Error in FastForest construction : Forest has " << tree_root_index.size() << " trees, which is not compatible with " << num_classes << "classes!" << std::endl;
            }
        }


        ////////////////////////////////////////////////////////////////////////
        /// \brief
        /// \param input
        /// \return
        inline
        float xhquant_xgboost_regressor::forward(const float *input) const
        {
            float out = 0.;
            for (int index : tree_root_index)
            {
                do
                {
                    int r = tree_right_indices[index];
                    int l = tree_left_indices[index];
                    index = input[tree_cut_indices[index]] > tree_cut_values[index] ? r : l;
                } while (index > 0);

                out += tree_responses[-index];
            }

            return out + base_score;
        }

        ////////////////////////////////////////////////////////////////////////
        /// \brief
        /// \param s
        /// \param delimiter
        /// \return
        inline
        std::vector<std::string> xhquant_xgboost_regressor::split(const std::string &str, const std::string &delimiter)
        {
            std::vector<std::string> elements;
            size_t pos = 0;
            size_t len = str.length();
            size_t delimiter_len = delimiter.length();
            if (delimiter_len == 0) return elements;
            while (pos < len)
            {
                int find_pos = str.find(delimiter, pos);
                if (find_pos < 0)
                {
                    elements.push_back(str.substr(pos, len - pos));
                    break;
                }
                elements.push_back(str.substr(pos, find_pos - pos));
                pos = find_pos + delimiter_len;
            }
            return elements;
        }


        ////////////////////////////////////////////////////////////////////////
        /// \brief
        /// \param s
        /// \return
        inline
        bool xhquant_xgboost_regressor::is_integer(const std::string &s)
        {
            if (s.empty() || ((!isdigit(s[0])) && (s[0] != '-') && (s[0] != '+')))
            {
                return false;
            }

            char *p = nullptr;
            strtol(s.c_str(), &p, 10);

            return (*p == 0);
        }


        ////////////////////////////////////////////////////////////////////////
        /// \brief
        /// \tparam data_type
        /// \param str
        /// \param substr
        /// \return
        template<class data_type>
        inline
        xhquant_xgboost_regressor::numeric_after_substr_output<data_type> xhquant_xgboost_regressor::numeric_after_substr(const std::string &str, const std::string &substr)
        {
            numeric_after_substr_output<data_type> output;
            output.rest = str;

            std::size_t found = str.find(substr);
            if (found != std::string::npos)
            {
                output.found = true;
                std::stringstream ss(str.substr(found + substr.size(), str.size() - found + substr.size()));
                ss >> output.value;
                if (!ss.fail())
                {
                    output.failed = false;
                    output.rest = ss.str();
                }
            }
            return output;
        }


        ////////////////////////////////////////////////////////////////////////
        /// \brief
        /// \param num_previous_nodes
        /// \param num_previous_leaves
        /// \param node_indices
        /// \param leaf_indices
        /// \param trees_skipped
        inline
        void xhquant_xgboost_regressor::terminate_tree(int &num_previous_nodes, int &num_previous_leaves, std::map<int, int> &node_indices, std::map<int, int> &leaf_indices, int &trees_skipped)
        {
            correct_indices(tree_right_indices.begin() + num_previous_nodes, tree_right_indices.end(), node_indices, leaf_indices);
            correct_indices(tree_left_indices.begin() + num_previous_nodes, tree_left_indices.end(), node_indices, leaf_indices);

            if (num_previous_nodes != static_cast<int>(tree_cut_values.size()))
            {
                tree_numbers.push_back(tree_root_index.size() + trees_skipped);
                tree_root_index.push_back(num_previous_nodes);
            }
            else
            {
                int treeNumbers = static_cast<int>(tree_root_index.size()) + trees_skipped;
                ++trees_skipped;

                tree_base_responses[treeNumbers % tree_base_responses.size()] += tree_responses.back();
                tree_responses.pop_back();
            }

            node_indices.clear();
            leaf_indices.clear();
            num_previous_nodes = tree_cut_values.size();
            num_previous_leaves = tree_responses.size();
        }


        ////////////////////////////////////////////////////////////////////////
        /// \brief
        /// \param begin
        /// \param end
        /// \param node_indices
        /// \param leaf_indices
        inline
        void xhquant_xgboost_regressor::correct_indices(std::vector<int>::iterator begin, std::vector<int>::iterator end, const std::map<int, int> &node_indices, const std::map<int, int> &leaf_indices)
        {
            for (auto iter = begin; iter != end; ++iter)
            {
                if (node_indices.count(*iter))
                {
                    *iter = node_indices.at(*iter);
                }
                else if (leaf_indices.count(*iter))
                {
                    *iter = -leaf_indices.at(*iter);
                }
                else
                {
                    throw std::runtime_error("something is wrong in the node structure");
                }
            }
        }
    }
}

#endif //XHQUANT_XGBOOST_REGRESSOR_HPP