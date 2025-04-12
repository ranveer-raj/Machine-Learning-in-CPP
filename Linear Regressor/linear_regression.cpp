#include<bits/stdc++.h>
using namespace std;

class DataFrame
{
private:
    map<string, vector<double>> data_raw;
    vector<vector<double>> data_scale_and_reorder;
    vector<vector<double>> data_transposed_scaled;
    vector<vector<double>> data_train_set;
    vector<vector<double>> data_test_set;

    double check_min_val(const vector<double>& data)
    {
        double min_val = numeric_limits<double>::max();
        bool found_valid = false;

        for (const auto& value : data)
        {
            if (!isnan(value))
            {
                min_val = min(min_val, value);
                found_valid = true;
            }
        }

        return found_valid ? min_val : NAN;
    }

    double check_max_val(const vector<double>& data)
    {
        double max_val = numeric_limits<double>::lowest();
        bool found_valid = false;

        for (const auto& value : data)
        {
            if (!isnan(value))
            {
                max_val = max(max_val, value);
                found_valid = true;
            }
        }

        return found_valid ? max_val : NAN;
    }

    string trim(const string& str)
    {
        size_t start = str.find_first_not_of(' ');
        size_t end = str.find_last_not_of(' ');
        if (start == string::npos || end == string::npos)
        {
            return "";
        }
        return str.substr(start, end - start + 1);
    }

    string check_dtypes(const vector<string> &cur_data)
    {
        if (cur_data.empty())
        {
            return "NULL";
        }
        try
        {
            stold(cur_data[0]);
            return "continuous";
        }
        catch (const invalid_argument&)
        {
            return "categorical";
        }
        catch (const out_of_range&)
        {
            return "continuous";
        }
    }

    vector<double> convert_continuous(const vector<string> &cur_data)
    {
        int m = cur_data.size();
        vector<double> data(m);
        for(int i=0;i<m;i++)
        {
            try
            {
                data[i] = stold(cur_data[i]);
            }
            catch (const invalid_argument&)
            {
                data[i] = NAN;
            }
            catch (const out_of_range&)
            {
                data[i] = NAN;
            }
        }
        return data;
    }

    pair<vector<int> , map<int, string>> convert_categorical(const vector<string> &cur_data)
    {
        int m = cur_data.size();
        vector<int> data(m);
        set<string> st;
        for(int i=0;i<m;i++)
        {
            st.insert(cur_data[i]);
        }
        int x = 1;
        map<int, string> num_to_cat;
        map<string, int> cat_to_num;
        for(const auto& val : st)
        {
            cat_to_num[val] = x;
            num_to_cat[x] = val;
            x = x+1;
        }
        for(int i=0;i<m;i++)
        {
            data[i] = cat_to_num[cur_data[i]];
        }
        return make_pair(data,num_to_cat);
    }

    map<string, vector<double>> one_hot_encoding(const vector<int> &data,const map<int, string> &num_to_col,const string &col_name)
    {
        int m = data.size();
        map<string, vector<double>> df;
        for(const auto& [val, sub_col_name] : num_to_col)
        {
            if(val == 1)
            {
                continue;
            }
            string new_col_name = col_name + " " + sub_col_name;
            vector<double> arr(m);
            for(int i=0;i<m;i++)
            {
                if(data[i] == val)
                {
                    arr[i] = 1.0;
                }
                else
                {
                    arr[i] = 0.0;
                }
            }
            df[new_col_name] = arr;
        }
        return df;
    }

    void readData()
    {
        map<string, vector<double>> df;
        vector<vector<string>> data;
        string line;
        while (getline(cin, line))
        {
            stringstream lineStream(line); // Create a stringstream for splitting
            string cell;
            vector<string> row;
            while (getline(lineStream, cell, ',')) 
            {
                row.push_back(trim(cell));
            }
            data.push_back(row);
        }
        map<string, string> dtypes;
        map<string, map<int, string>> num_to_cat;
        int n = data[0].size(); // number of variables
        int m = data.size()-1; // number of features
        for(int j=0;j<n;j++)
        {
            string col_name = data[0][j];
            vector<string> cur_data(m);
            for(int i=1;i<=m;i++)
            {
                cur_data[i-1] = data[i][j];
            }
            dtypes[col_name] = check_dtypes(cur_data);
            if(dtypes[col_name] == "continuous")
            {
                df[col_name] = convert_continuous(cur_data);
            }
            else
            {
                pair<vector<int> , map<int, string>> result = convert_categorical(cur_data);
                map<string, vector<double>> encoded_result = one_hot_encoding(result.first, result.second, col_name);
                for (const auto& [value1, value2] : encoded_result)
                {
                    df[value1] = value2;
                }
            }
        }
        data_raw = df;
    }

    vector<double> standard_scaler(const vector<double> &arr)
    {
        vector<double> X = arr;
        int n = X.size();
        if (n == 0) return {};
        for (auto& x : X)
        {
            if (isnan(x)) x = 0.0;
        }
        double mean = accumulate(X.begin(), X.end(), 0.0) / n;
        double variance = 0.0;
        for (const auto &x : X)
        {
            variance += ((x - mean)*(x - mean));
        }
        double std_dev = sqrt(variance / n);
        vector<double> X_scaled(n);
        for (int i = 0; i < n; ++i) {
            if (std_dev != 0)
                X_scaled[i] = (X[i] - mean) / std_dev;
            else
                X_scaled[i] = 0.0; // Handle case where all values are the same
        }
        return X_scaled;
    }

    void scale_and_reorder_data(const map<string, vector<double>> &data, const string &target)
    {
        vector<vector<double>> df;
        vector<double> target_data;
        for(const auto& [val, cur_data] : data)
        {
            if(val==target)
            {
                target_data = cur_data;
            }
            else
            {
                df.push_back(standard_scaler(cur_data));
            }
        }
        df.push_back(target_data);
        data_scale_and_reorder = df;
    }
    
    bool are_linearly_independent(const vector<double> &vec1, const vector<double> &vec2, double tolerance = 1e-10)
    {
        int n = vec1.size();
        if (vec2.size() != n)
        {
            throw invalid_argument("Vectors must have the same size.");
        }
        double ratio = 0.0;
        bool ratio_set = false;
        for (int i = 0; i < n; ++i)
        {
            if (abs(vec2[i]) > tolerance)
            {
                double current_ratio = vec1[i] / vec2[i];
                if (!ratio_set)
                {
                    ratio = current_ratio;
                    ratio_set = true;
                }
                else
                {
                    if (abs(current_ratio - ratio) > tolerance)
                    {
                        return true; // Vectors are independent
                    }
                }
            }
            else if (abs(vec1[i]) > tolerance)
            {
                return true; // vec2[i] is zero but vec1[i] is not
            }
        }
        return false; // Vectors are dependent
    }

    vector<vector<double>> remove_linear_dependent_variable(const vector<vector<double>> &data)
    {
        set<int> col_to_remove;
        int n = data[0].size();
        int m = data.size()-1; // subtracted 1 to remove target variable when checking linear dependency
        for(int j=0;j<m;j++)
        {
            vector<double> vec1(n);
            for(int i=0;i<n;i++)
            {
                vec1[i] = data[j][i];
            }
            for(int k=j+1;k<m;k++)
            {
                vector<double> vec2(n);
                for(int i=0;i<n;i++)
                {
                    vec2[i] = data[k][i];
                }
                if(!are_linearly_independent(vec1, vec2))
                {
                    cout<<j<<" "<<k<<endl;
                    col_to_remove.insert(k);
                }
            }
        }
        vector<vector<double>> df;
        for(int j=0;j<m;j++)
        {
            if(col_to_remove.find(j)==col_to_remove.end())
            {
                df.push_back(data[j]);
            }
        }
        return data;
    }

    void transpose(const vector<vector<double>> &data)
    {
        int n = data.size();
        int m = data[0].size();
        vector<vector<double>> df(m,vector<double>(n));
        for(int i=0;i<m;i++)
        {
            for(int j=0;j<n;j++)
            {
                df[i][j] = data[j][i];
            }
        }
        data_transposed_scaled = df;
    }

    set<int> split_data_train_set(const int m,const int m_train,const int seed)
    {
        vector<int> data(m);
        for (int i = 0; i < m; i++) {
            data[i] = i;
        }
        mt19937 gen(seed);
        shuffle(data.begin(), data.end(), gen);
        set<int> train(data.begin(), data.begin() + m_train);
        return train;
    }

    void train_and_test_split(const vector<vector<double>> &data, const double test_size = 0.25, const int seed = 42)
    {
        int m = data.size();
        int n = data[0].size();
        int m_test = (int)(m*test_size);
        int m_train = m-m_test;
        vector<vector<double>> df_train(m_train,vector<double>(n));
        vector<vector<double>> df_test(m_test,vector<double>(n));
        set<int> st_train = split_data_train_set(m, m_train, seed);
        for(int i=0,i_train=0,i_test=0; i<m; i++)
        {
            if(st_train.find(i)!=st_train.end())
            {
                for(int j=0;j<n;j++)
                {
                    df_train[i_train][j] = data[i][j];
                }
                i_train++;
            }
            else
            {
                for(int j=0;j<n;j++)
                {
                    df_test[i_test][j] = data[i][j];
                }
                i_test++;
            }
        }
        data_train_set = df_train;
        data_test_set = df_test;
    }
public:
    DataFrame(){}
    void read_data()
    {
        readData();
    }
    void scale_and_reorder(const string target)
    {
        scale_and_reorder_data(data_raw, target);
        transpose(remove_linear_dependent_variable(data_scale_and_reorder));
    }
    void train_test_split()
    {
        train_and_test_split(data_transposed_scaled);
    }
    vector<vector<double>> get_train()
    {
        return data_train_set;
    }
    vector<vector<double>> get_test()
    {
        return data_test_set;
    }
};

class LinearRegressor
{
private:
    vector<vector<double>> df_train, df_test;
    string algo_name;
    vector<double> theta;
    void batch_gradient_descent()
    {

    }
    void mini_batch_gradient_descent()
    {

    }
    void stochastic_gradient_descent()
    {

    }
    void fit_transform()
    {
        if (algo_name == "batch")
        {
            batch_gradient_descent();
        }
        else if (algo_name == "mini-batch")
        {
            mini_batch_gradient_descent();
        }
        else if (algo_name == "stochastic")
        {
            stochastic_gradient_descent();
        }
        else
        {
            throw invalid_argument("Unknown algorithm name: " + algo_name);
        }
    }
public:
    LinearRegressor() {}

    void fit(const string algo, const vector<vector<double>> train, const vector<vector<double>> test)
    {
        algo_name = algo;
        df_train = train;
        df_test = test;
        fit_transform();
    }
};

signed main()
{
    #ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
    freopen("error.txt", "w", stderr);
    #endif
    ios_base::sync_with_stdio(0);
    cin.tie(0);cout.tie(0);

    DataFrame data;
    data.read_data();
    data.scale_and_reorder("Life expectancy"); // pass target variable
    data.train_test_split();

    LinearRegressor model1, model2, model3;
    model1.fit("batch", data.get_train(), data.get_test());
    model2.fit("mini-batch", data.get_train(), data.get_test());
    model3.fit("stochastic", data.get_train(), data.get_test());

    return 0;
}