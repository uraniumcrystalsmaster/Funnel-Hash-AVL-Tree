//
// Created by urani on 9/18/2025.
//

#ifndef AVL_TREE_H
#define AVL_TREE_H
#include <stack>
//#include <unordered_map>
#include <vector>
//#include "Elastic_Hash_Map.h" //doesn't work
#include "Funnel_Hash_Map.h"
template <typename Key, typename Value>
class AVL_Tree{
    public:
        struct NodeProps {
            Key key;
            Value value;
            Key left_key = Key{};
            Key right_key = Key{};
            Key parent_key = Key{};
            int height = 0;
        };
    private:
        //std::unordered_map<Key, NodeProps> umap; // expected worst case O(N)
        //Elastic_Hash_Map<Key, NodeProps> umap;   // expected worst case O(1)
        Funnel_Hash_Map<Key, NodeProps> umap;     // expected worst case O(1)
        Key root_key = Key{};
    public:
        explicit AVL_Tree(size_t N,double load_factor = 0.75) : umap(N) {
            //umap.reserve(N);
        }

        size_t size() {
            return umap.size();
        }

        void balance_tree(const Key& key, bool deletion = false) {
            updateHeight(key);
            NodeProps& curr_node = umap.find(key)->second;
            int curr_node_bal_fac = getHeight(curr_node.left_key) - getHeight(curr_node.right_key);
            if(curr_node_bal_fac <= -2) {
                NodeProps& right_child = umap.find(curr_node.right_key)->second;
                int right_child_bal_fac = getHeight(right_child.left_key) - getHeight(right_child.right_key);
                if(right_child_bal_fac > 0) {
                    right_left_rotate(key);
                }
                else {
                    left_rotate(key);
                }
            }
            else if(curr_node_bal_fac >= 2) {
                NodeProps& left_child = umap.find(curr_node.left_key)->second;
                int left_child_bal_fac = getHeight(left_child.left_key) - getHeight(left_child.right_key);
                if(left_child_bal_fac < 0) {
                    left_right_rotate(key);
                }
                else {
                    right_rotate(key);
                }
            }
        }

        int getHeight(const Key& key) {
            if(umap.find(key) == umap.end()) {
                return -1;
            }
            return umap.find(key)->second.height;
        }

        void updateHeight(const Key& key){
            NodeProps& curr_node = umap.find(key)->second;
            curr_node.height = 1 + std::max(getHeight(curr_node.left_key), getHeight(curr_node.right_key));
        }

        void left_rotate(const Key& parent_key) {
            NodeProps& parent = umap.find(parent_key)->second;
            Key child_key = parent.right_key;
            //return if no right child
            if(parent.right_key == Key{}) {
                return;
            }
            NodeProps& child = umap.find(child_key)->second;
            Key grandparent_key = parent.parent_key;
            Key detached_subtree_key = child.left_key;
            child.parent_key = grandparent_key;
            if (grandparent_key == Key{}) {
                this->root_key = child_key;
            }
            else {
                NodeProps& grandparent = umap.find(grandparent_key)->second;
                if (grandparent.left_key == parent_key) {
                    grandparent.left_key = child_key;
                }
                else {
                    grandparent.right_key = child_key;
                }
            }
            parent.right_key = detached_subtree_key;
            if (detached_subtree_key != Key{}) {
                umap.find(detached_subtree_key)->second.parent_key = parent_key;
            }
            child.left_key = parent_key;
            parent.parent_key = child_key;


            updateHeight(parent_key);
            updateHeight(child_key);
        }

        void right_rotate(const Key& parent_key) {
            NodeProps& parent = umap.find(parent_key)->second;
            Key child_key = parent.left_key;
            //return if no left child
            if(parent.left_key == Key{}) {
                return;
            }
            NodeProps& child = umap.find(child_key)->second;
            Key grandparent_key = parent.parent_key;
            Key detached_subtree_key = child.right_key;
            child.parent_key = grandparent_key;
            if (grandparent_key == Key{}) {
                this->root_key = child_key;
            }
            else {
                NodeProps& grandparent = umap.find(grandparent_key)->second;
                if (grandparent.right_key == parent_key) {
                    grandparent.right_key = child_key;
                }
                else {
                    grandparent.left_key = child_key;
                }
            }
            parent.left_key = detached_subtree_key;
            if (detached_subtree_key != Key{}) {
                umap.find(detached_subtree_key)->second.parent_key = parent_key;
            }
            child.right_key = parent_key;
            parent.parent_key = child_key;


            updateHeight(parent_key);
            updateHeight(child_key);
        }

        void left_right_rotate(const Key& key) {
            Key child_key = umap.find(key)->second.left_key;
            left_rotate(child_key);
            right_rotate(key);
        }

        void right_left_rotate(const Key& key) {
            Key child_key = umap.find(key)->second.right_key;
            right_rotate(child_key);
            left_rotate(key);
        }

        void insert(Key key, Value value) {
            if(umap.find(key) != umap.end()) {
                std::cout << "unsuccessful" << std::endl;
                return;
            }
            // Reformat input
            NodeProps node_props;
            node_props.key = key;
            node_props.value = value;
            std::pair<Key, NodeProps> map_pair(key, node_props);

            // Start insertion
            if(this->root_key == Key{}) {
                umap.emplace(map_pair);
                this->root_key = key;
                //std::cerr << "Insertion of root with key '" << key << "'\n";
                std::cout << "successful" << std::endl;
                return;
            }
            Key nav_key = this->root_key;
            while(true) {
                if(key < nav_key) {
                    if(umap.find(nav_key)->second.left_key == Key{}) {
                        umap.emplace(map_pair);
                        umap.find(nav_key)->second.left_key = key;
                        umap.find(key)->second.parent_key = nav_key;
                        Key balance_nav_key = key;
                        while(true){
                            balance_nav_key =  umap.find(balance_nav_key)->second.parent_key;
                            if(balance_nav_key == Key{}) {
                                break;
                            }
                            balance_tree(balance_nav_key);
                        }
                        //std::cerr << "Insertion with nav_key '" << nav_key << "' and key '" << key << "'\n";
                        std::cout << "successful" << std::endl;
                        return;
                    }
                    nav_key = umap.find(nav_key)->second.left_key;
                }
                else {
                    if(umap.find(nav_key)->second.right_key == Key{}) {
                        umap.emplace(map_pair);
                        umap.find(nav_key)->second.right_key = key;
                        umap.find(key)->second.parent_key = nav_key;
                        Key balance_nav_key = key;
                        while(true){
                            balance_nav_key =  umap.find(balance_nav_key)->second.parent_key;
                            if(balance_nav_key == Key{}) {
                                break;
                            }
                            balance_tree(balance_nav_key);
                        }
                        //std::cerr << "Insertion with nav_key '" << nav_key << "' and key '" << key << "'\n";
                        std::cout << "successful" << std::endl;
                        return;
                    }
                    nav_key = umap.find(nav_key)->second.right_key;
                }
            }
        }

        void remove(Key key) {
            //if key doesn't exist
            if(umap.find(key) == umap.end()) {
                std::cout << "unsuccessful" << std::endl;
                //std::cerr << "failed to erase key: " << key << std::endl;
                return;
            }

            Key balance_nav_key;

            //// 1. unlink node
            NodeProps& node_to_remove = umap.find(key)->second;

            //node without children case
            if(node_to_remove.left_key == Key{} and node_to_remove.right_key == Key{}) {
                balance_nav_key = node_to_remove.parent_key;
                if(key != this->root_key) {
                    NodeProps& node_to_remove_parent = umap.find(node_to_remove.parent_key)->second;
                    if(node_to_remove_parent.left_key == key) {
                        node_to_remove_parent.left_key = Key{};
                    }
                    else {
                        node_to_remove_parent.right_key = Key{};
                    }
                }
                else {
                    this->root_key = Key{};
                }
            }
            //node with two children case
            else if(node_to_remove.left_key != Key{} and node_to_remove.right_key != Key{}) {
                //find inorder successor
                Key inorder_successor_key = node_to_remove.right_key;
                while(umap.find(inorder_successor_key)->second.left_key != Key{}) {
                    inorder_successor_key = umap.find(inorder_successor_key)->second.left_key;
                }
                NodeProps& inorder_successor = umap.find(inorder_successor_key)->second;

                //reassign links - easy case
                if(inorder_successor.key == node_to_remove.right_key) {
                    inorder_successor.left_key = node_to_remove.left_key;
                    inorder_successor.parent_key = node_to_remove.parent_key;
                    if(inorder_successor.right_key != Key{}) {
                        NodeProps& successor_right_child = umap.find(inorder_successor.right_key)->second;
                        successor_right_child.parent_key = inorder_successor.key;
                    }

                    NodeProps& node_to_remove_left_child = umap.find(node_to_remove.left_key)->second;
                    node_to_remove_left_child.parent_key = inorder_successor.key;

                    if(node_to_remove.parent_key != Key{}) {
                        NodeProps& node_to_remove_parent = umap.find(node_to_remove.parent_key)->second;
                        if(node_to_remove_parent.left_key == key) {
                            node_to_remove_parent.left_key = inorder_successor.key;
                        }
                        else {
                            node_to_remove_parent.right_key = inorder_successor.key;
                        }
                    }
                    balance_nav_key = inorder_successor_key;
                }
                //reassign links - hard case
                else {
                    NodeProps& successor_parent = umap.find(inorder_successor.parent_key)->second;
                    Key successor_right_key = inorder_successor.right_key;
                    successor_parent.left_key = successor_right_key;
                    if (successor_right_key != Key{}) {
                        umap.find(successor_right_key)->second.parent_key = successor_parent.key;
                    }
                    inorder_successor.right_key = node_to_remove.right_key;
                    umap.find(node_to_remove.right_key)->second.parent_key = inorder_successor_key;

                    Key parent_key = node_to_remove.parent_key;
                    inorder_successor.parent_key = parent_key;
                    if (parent_key == Key{}) {
                        this->root_key = inorder_successor_key;
                    }
                    else {
                        NodeProps& parent = umap.find(parent_key)->second;
                        if (parent.left_key == key) {
                            parent.left_key = inorder_successor_key;
                        }
                        else {
                            parent.right_key = inorder_successor_key;
                        }
                    }

                    inorder_successor.left_key = node_to_remove.left_key;
                    umap.find(node_to_remove.left_key)->second.parent_key = inorder_successor_key;
                    balance_nav_key = inorder_successor.parent_key;
                }

                if(this->root_key == key) {
                    this->root_key = inorder_successor_key;
                }
            }
            //node with one child case
            else {
                balance_nav_key = node_to_remove.parent_key;
                if(key != this->root_key) {
                    NodeProps& node_to_remove_parent = umap.find(node_to_remove.parent_key)->second;
                    if(node_to_remove_parent.left_key == key) {
                        if(node_to_remove.left_key != Key{}) {
                            //handle left line
                            NodeProps& node_to_remove_child = umap.find(node_to_remove.left_key)->second;
                            node_to_remove_child.parent_key = node_to_remove.parent_key;
                            node_to_remove_parent.left_key = node_to_remove.left_key;
                        }
                        else {
                            //handle left right zig zag
                            NodeProps& node_to_remove_child = umap.find(node_to_remove.right_key)->second;
                            node_to_remove_child.parent_key = node_to_remove.parent_key;
                            node_to_remove_parent.left_key = node_to_remove.right_key;
                        }
                    }
                    else {
                        if(node_to_remove.right_key != Key{}) {
                            //handle right line
                            NodeProps& node_to_remove_child = umap.find(node_to_remove.right_key)->second;
                            node_to_remove_child.parent_key = node_to_remove.parent_key;
                            node_to_remove_parent.right_key = node_to_remove.right_key;
                        }
                        else {
                            //handle right zig zag
                            NodeProps& node_to_remove_child = umap.find(node_to_remove.left_key)->second;
                            node_to_remove_child.parent_key = node_to_remove.parent_key;
                            node_to_remove_parent.right_key = node_to_remove.left_key;
                        }
                    }
                }
                else {
                    if(node_to_remove.left_key != Key{}) {
                        this->root_key = node_to_remove.left_key;
                        umap.find(node_to_remove.left_key)->second.parent_key = Key{};
                    }
                    else {
                        this->root_key = node_to_remove.right_key;
                        umap.find(node_to_remove.right_key)->second.parent_key = Key{};
                    }
                }
            }
            //// 2. delete unlinked node
            umap.erase(key);
            //std::cerr << "erased key: " << key << std::endl;

            //// 3. rebalance tree
            while(balance_nav_key != Key{}) {
                balance_tree(balance_nav_key);
                balance_nav_key = umap.find(balance_nav_key)->second.parent_key;
            }

            //// 4. print success (optional)
            std::cout << "successful" << std::endl;
        }

        void printPreorder() {
            if (this->root_key == Key{}) {
                return;
            }

            std::stack<Key> keychain;
            keychain.push(this->root_key);
            bool first = true;

            while (!keychain.empty()) {
                Key nav_key = keychain.top();
                keychain.pop();

                if (!first) {
                    std::cout << ", ";
                }
                std::cout << umap.find(nav_key)->second.value;
                first = false;

                NodeProps& nav_node = umap.find(nav_key)->second;

                if (nav_node.right_key != Key{}) {
                    keychain.push(nav_node.right_key);
                }

                if (nav_node.left_key != Key{}) {
                    keychain.push(nav_node.left_key);
                }
            }
            std::cout << std::endl;
        }

        void printInorder() {
            if (this->root_key == Key{}) {
                return;
            }

            std::stack<Key> keychain;
            Key nav_key = this->root_key;
            bool first_print = true;

            while (nav_key != Key{} or !keychain.empty()) {
                while (nav_key != Key{}) {
                    keychain.push(nav_key);
                    nav_key = umap.find(nav_key)->second.left_key;
                }

                nav_key = keychain.top();
                keychain.pop();

                if (!first_print) {
                    std::cout << ", ";
                }
                first_print = false;
                std::cout << umap.find(nav_key)->second.value;
                
                nav_key = umap.find(nav_key)->second.right_key;
            }
            std::cout << std::endl;
        }

        void printPostorder() {
            if (this->root_key == Key{}) {
                return;
            }

            std::stack<Key> keychain;
            std::stack<Key> keychain2;
            keychain.push(this->root_key);
            while (!keychain.empty()) {
                Key nav_key = keychain.top();
                keychain.pop();
                keychain2.push(nav_key);

                NodeProps& nav_node = umap.find(nav_key)->second;
                if (nav_node.left_key != Key{}) {
                    keychain.push(nav_node.left_key);
                }

                if (nav_node.right_key != Key{}) {
                    keychain.push(nav_node.right_key);
                }
            }

            bool first = true;
            while (!keychain2.empty()) {
                Key nav_key = keychain2.top();
                keychain2.pop();
                if (!first) {
                    std::cout << ", ";
                }
                std::cout << umap.find(nav_key)->second.value;
                first = false;
            }
            std::cout << std::endl;
        }

        Value* searchByKey(Key key) {
            if(umap.find(key) == umap.end()) {
                std::cout << "unsuccessful" << std::endl;
                return nullptr;
            }
            return &umap.find(key)->second.value;
        }

        void searchByKeyPrint(Key key) {
            if(umap.find(key) == umap.end()) {
                std::cout << "unsuccessful" << std::endl;
                return;
            }
            std::cout << umap.find(key)->second.value << std::endl;
        }

        std::vector<Key&> searchByValue(Value value) {
            std::vector<Key&> found_keys;
            if (this->root_key == Key{}) {
                return found_keys;
            }
            std::stack<Key> keychain;
            keychain.push(this->root_key);
            bool first = true;

            while (!keychain.empty()) {
                Key nav_key = keychain.top();
                keychain.pop();

                NodeProps& nav_node = umap.find(nav_key)->second;
                if(nav_node.value == value) {
                    if (!first) {
                        std::cout << ", ";
                    }
                    std::cout << nav_key;
                    found_keys.push_back(nav_key);
                    first = false;
                }

                if (umap.find(nav_node.right_key) != umap.end()) {
                    keychain.push(nav_node.right_key);
                }

                if (umap.find(nav_node.left_key) != umap.end()) {
                    keychain.push(nav_node.left_key);
                }
            }
            return found_keys;
        }

        void searchByValuePrint(Value value) {
            if (this->root_key == Key{}) {
                std::cout << "unsuccessful" << std::endl;
                return;
            }
            std::stack<Key> keychain;
            keychain.push(this->root_key);
            bool first = true;

            while (!keychain.empty()) {
                Key nav_key = keychain.top();
                keychain.pop();

                NodeProps& nav_node = umap.find(nav_key)->second;
                if(nav_node.value == value) {
                    std::cout << nav_key << std::endl;
                    first = false;
                }

                if (umap.find(nav_node.right_key) != umap.end()) {
                    keychain.push(nav_node.right_key);
                }

                if (umap.find(nav_node.left_key) != umap.end()) {
                    keychain.push(nav_node.left_key);
                }
            }
            if(first) {
                std::cout << "unsuccessful" << std::endl;
            }
        }

        void printLevelCount() {
            int level = 0;
            if(this->root_key == Key{}){
                std::cout << level << std::endl;
                return;
            }
            for(auto iter = umap.begin(); iter != umap.end(); ++iter) {
                if(level < iter->second.height){
                    level = iter->second.height;
                }
            }
            level++;
            std::cout<<level<<std::endl;
        }

        void removeInorder(size_t N) {
            if (this->root_key == Key{}) {
                return;
            }

            std::stack<Key> keychain;
            Key nav_key = this->root_key;
            size_t i = 0;
            while (nav_key != Key{} or !keychain.empty()) {
                while (nav_key != Key{}) {
                    keychain.push(nav_key);
                    nav_key = umap.find(nav_key)->second.left_key;
                }

                nav_key = keychain.top();
                keychain.pop();

                if(i == N) {
                    remove(nav_key);
                    return;
                }

                nav_key = umap.find(nav_key)->second.right_key;
                i++;
            }
            std::cout << "unsuccessful:" << std::endl;
        }

        void printRandom(bool only_occupied = false) {
            this->umap.printRandom(only_occupied);
        }

        //getters and setters
        Key getRootKey() {
            return this->root_key;
        }

        bool isEmpty() {
            if(this->root_key == Key{}) {
                return true;
            }
            return false;
        }
};
#endif //AVL_TREE_H
