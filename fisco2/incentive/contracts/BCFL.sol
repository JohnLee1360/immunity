// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0;

contract BCFL {
    // 账户角色
    address public arbiter;
    address public client1;
    address public client2;
    address public client3;

    // 用于记录质押分和激励分
    struct Client {
        uint256 stake_credit;
        uint256 incent_credit;
        bool isBlacklisted;
    }

    // 客户端的状态
    mapping(address => Client) public clients;

    // 事件
    event StakeDeposited(address indexed client, uint256 amount);
    event IncentiveRewarded(address indexed client, uint256 amount);
    event StakeDeducted(address indexed client, uint256 amount);
    event Blacklisted(address indexed client);

    modifier onlyArbiter() {
        require(msg.sender == arbiter, "Only arbiter can perform this action");
        _;
    }

    modifier onlyClient() {
        require(
            msg.sender == client1 || msg.sender == client2 || msg.sender == client3,
            "Only clients can perform this action"
        );
        _;
    }

    modifier notBlacklisted(address client) {
        require(!clients[client].isBlacklisted, "Client is blacklisted");
        _;
    }

    constructor(address _client1, address _client2, address _client3) {
        arbiter = msg.sender;
        client1 = _client1;
        client2 = _client2;
        client3 = _client3;

        // 初始化客户端状态
        clients[client1] = Client(0, 0, false);
        clients[client2] = Client(0, 0, false);
        clients[client3] = Client(0, 0, false);
    }

    // 客户端质押
    function depositStake() external onlyClient notBlacklisted(msg.sender) {
        require(clients[msg.sender].stake_credit >= 100, "Insufficient stake credit");
        clients[msg.sender].stake_credit -= 100;
        emit StakeDeposited(msg.sender, 100);
    }

    // 协调者奖励激励分
    function rewardIncentive(address client) external onlyArbiter notBlacklisted(client) {
        clients[client].incent_credit += 1;
        emit IncentiveRewarded(client, 1);
    }

    // 扣除质押分（检测到攻击行为）
    function deductStake(address client) external onlyArbiter notBlacklisted(client) {
        require(clients[client].stake_credit >= 200, "Insufficient stake to deduct");
        clients[client].stake_credit -= 200;
        emit StakeDeducted(client, 200);

        // 检查是否需要加入黑名单
        if (clients[client].stake_credit == 0) {
            clients[client].isBlacklisted = true;
            emit Blacklisted(client);
        }
    }

    // 查看黑名单状态
    function isBlacklisted(address client) external view returns (bool) {
        return clients[client].isBlacklisted;
    }

    // 初始化客户端质押分
    function initializeStake(address client, uint256 amount) external onlyArbiter {
        clients[client].stake_credit += amount;
    }

    // 初始化客户端激励分
    function initializeIncentive(address client, uint256 amount) external onlyArbiter {
        clients[client].incent_credit += amount;
    }
}
